import ast
import base64
import os
import tempfile
import uuid
from typing import AsyncIterable, Iterable, List, Optional

import google.generativeai as genai
import openai.types
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel
from vertexai.generative_models import (Content, FunctionDeclaration, GenerationConfig, GenerationResponse,
                                        GenerativeModel, Part, Tool)

app = FastAPI()
router = APIRouter()
genai.configure(api_key=os.environ['GEMINI_API_KEY'])


@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()
    print("REQUEST BODY: ", body)
    return await call_next(request)


class Message(BaseModel):
    role: str
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None


class Delta(Message):
    tool_calls: list[dict] | None = None


class Question(BaseModel):
    description: str
    type: str


class Parameters(BaseModel):
    type: str
    # properties: Properties
    properties: dict


class Function(BaseModel):
    name: str
    description: str
    parameters: dict


class OAITool(BaseModel):
    type: str
    function: Function


class BetaOAIParameters(BaseModel):
    type: str
    properties: dict | None = None
    required: list[str] | None = None


class BetaOAIFunction(BaseModel):
    name: str
    description: str
    parameters: BetaOAIParameters


class BetaOAITool(BaseModel):
    type: str | str = "function"
    function: BetaOAIFunction


class BetaOAIToolCall(BetaOAITool):
    id: str | None = None

    def generate_id(self) -> str:
        return self.id if self.id else generate_id("call")


class BetaGeminiParameters(BaseModel):
    type: str
    properties: dict | None = None
    required: list[str] | None = None


class BetaGeminiFunction(BaseModel):
    name: str
    description: str
    parameters: BetaGeminiParameters

    def convert_to_sdk_function_declaration(self) -> FunctionDeclaration:
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters.dict()
        )


class BetaGeminiTool(BaseModel):
    function_declarations: list[BetaGeminiFunction] | None = None

    def convert_to_sdk_tool(self) -> Tool:
        functions: list[FunctionDeclaration] = []
        for function in self.function_declarations:
            functions.append(function.convert_to_sdk_function_declaration())
        return Tool(functions)


def map_oai_to_gemini_function(oai_function: BetaOAIFunction) -> BetaGeminiFunction:
    return BetaGeminiFunction(
        name=oai_function.name,
        description=oai_function.description,
        parameters=BetaGeminiParameters(
            type=oai_function.parameters.type,
            properties=oai_function.parameters.properties
        )
    )


def map_gemini_to_oai_function(gemini_function: BetaGeminiFunction) -> BetaOAIFunction:
    oai_function = BetaOAIFunction()
    oai_function.name = gemini_function.name
    oai_function.description = gemini_function.description
    oai_function.parameters = gemini_function.parameters
    return oai_function


class OAICompletionRequest(BaseModel):
    model: str
    messages: List[Message] | None = None
    max_tokens: int | None = None
    stream: bool | None = False
    seed: float | None = None
    tools: List[OAITool] | None = None
    tool_choice: List[OAITool] | Optional[str] | None = None
    top_k: int | None = None
    top_p: float | None = None
    temperature: float | None = None


class TopLogprob(BaseModel):
    token: str
    logprob: int | None = None
    bytes: List[int]


class OAIContent(BaseModel):
    token: str | None = None
    logprob: Optional[int] = None
    bytes: Optional[List[int]] = None
    top_logprobs: Optional[List[TopLogprob]] = None


class Logprobs(BaseModel):
    content: List[OAIContent] | None = None


class Choice(BaseModel):
    finish_reason: str
    index: int
    delta: Delta
    logprobs: Logprobs


class ToolCall(BaseModel):
    id: str
    type: str
    function: Function


class RespChoice(BaseModel):
    finish_reason: str | None = None
    index: int
    delta: Delta
    logprobs: Logprobs
    tool_calls: List[ToolCall] | None = None


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class OAICompletionResponse(BaseModel):
    id: str
    choices: List[RespChoice]
    created: int = None
    model: str
    system_fingerprint: str
    object: str
    usage: Usage | None = None
    temperature: float | None = None


def init_cache_dir() -> str:
    tmp = tempfile.mkdtemp()
    print(type(tmp))
    return tmp


cache_dir = init_cache_dir()


def put_cache(dir: str, key: str, value: str) -> str:
    path = os.path.join(dir, key)
    with open(path, "w") as f:
        f.write(str(value))
    return path


def get_cache(path: str) -> str:
    with open(path) as f:
        return f.read()


def calculate_cache_key(value: str) -> str:
    value = bytes(value, 'utf-8')
    return base64.urlsafe_b64encode(value).decode('utf-8')


@app.get("/models")
def list_models() -> JSONResponse:
    models = []
    for model in genai.list_models():
        model = openai.types.Model(id=model.name.removeprefix("models/"), created=0, object="model", owned_by="system")
        models.append(model)

    models = {"data": models}
    models_json = jsonable_encoder(models)
    return JSONResponse(content=models_json)


@app.post("/chat/completions", response_model_exclude_none=True, response_model_exclude_unset=True,
          response_model=OAICompletionResponse)
async def chat_completion(request: Request, data: OAICompletionRequest):
    print("REQUESTBODY: ", await request.body())

    # print("GPTSCRIPT INPUT: ", data)
    gemini_tools: list[Tool] | None = None
    if data.tools is not None:
        gemini_functions: list[BetaGeminiFunction] = []
        for tool in data.tools:
            # TODO: REMOVE THIS MAPPING
            tool = BetaOAITool(
                type=tool.type,
                function=BetaOAIFunction(
                    name=tool.function.name,
                    description=tool.function.description,
                    parameters=BetaOAIParameters(
                        type=tool.function.parameters["type"],
                        properties=tool.function.parameters["properties"],
                    )
                )
            )
            gemini_function = map_oai_to_gemini_function(tool.function)
            gemini_functions.append(gemini_function)

        gemini_tools = [BetaGeminiTool(function_declarations=gemini_functions).convert_to_sdk_tool()]

    user_parts: list[Part] = []
    model_parts: list[Part] = []
    function_parts: list[Part] = []

    for message in data.messages:
        if message.content:
            part = Part.from_text(message.content)
            # print("PART FROM TEXT: ", part)

        # print("TOOL CALL ID: ", message.tool_call_id)
        if message.tool_call_id:
            decoded_tool_call_id = base64.urlsafe_b64decode(message.tool_call_id).decode('utf-8')
            cache_value = get_cache(decoded_tool_call_id)
            print("CACHE VALUE: ", cache_value)
            value = ast.literal_eval(cache_value)
            part = Part.from_function_response(name=value["name"],
                                               response={
                                                   "name": value["name"],
                                                   "content": message.content
                                               })
            # part = Part.from_function_response(name=value["name"],
            #                                    response={
            #                                        "name": value["name"],
            #                                        "content": str(value["args"])
            #                                    })
            # print("PART FROM FUNCTION_RESPONSE: ", part)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                # role for these should always be 'model'
                print("IN TOOL CALL, PRINT ARGS: ", ast.literal_eval(tool_call["function"]["arguments"]))

                args: dict = {"fields": {}}
                for key, value in ast.literal_eval(tool_call["function"]["arguments"]):
                    args["fields"]["key"] = key
                    args["fields"]["value"] = {"number_value": value}

                part = Part.from_dict({
                    "function_call": {
                        "name": tool_call["function"]["name"],
                        "args": args
                        # "args": ast.literal_eval(tool_call["function"]["arguments"])
                        # "args": {
                        #     # "question": "how are you?"
                        #     "fields": {
                        #         "key": "number",
                        #         "value": {
                        #             "number_value": 3
                        #         }
                        #     }
                        # }
                    }
                }
                )

        role: str
        match message.role:
            case "system":
                role = "user"
            case "user":
                role = "user"
            case "assistant":
                role = "model"
            case "model":
                role = "model"
            case "tool":
                role = "function"
            case _:
                role = "user"

        match role:
            case "user":
                user_parts.append(part)
            case "model":
                model_parts.append(part)
            case "function":
                function_parts.append(part)

    all_content: list[Content] = []

    if user_parts:
        all_content.append(Content(
            role="user",
            parts=user_parts
        ))

    if model_parts:
        all_content.append(Content(
            role="model",
            parts=model_parts
        ))

    if function_parts:
        all_content.append(Content(
            role="function",
            parts=function_parts
        ))

    print("ALL CONTENT: ", all_content)

    model = GenerativeModel("gemini-pro", tools=gemini_tools)
    print("CALLING GENERATE CONTENT")
    response = model.generate_content(all_content, tools=gemini_tools, stream=data.stream,
                                      generation_config=GenerationConfig(
                                          temperature=data.temperature,
                                          top_k=data.top_k,
                                          top_p=data.top_p,
                                          max_output_tokens=data.max_tokens,
                                      ))

    return StreamingResponse(async_chunk(response), media_type="application/x-ndjson")
    # return JSONResponse(content=response_json)


async def async_chunk(chunks: Iterable[GenerationResponse]) -> AsyncIterable[str]:
    # print("CHUNKS FROM GEMINI: ")
    for chunk in chunks:
        #         print(chunk)
        chunk = map_gemini_to_oai_response(chunk)
        yield "data: " + chunk.json() + "\n\n"


def generate_id(type: str = "chatcmpl") -> str:
    return type + "-" + str(uuid.uuid4())


def map_gemini_to_oai_response(response: GenerationResponse) -> OAICompletionResponse:
    choices = []
    print("PRINTING GEMINI RESPONSE AS DICT: ", response.to_dict())
    response = response.to_dict()
    for idx, item in enumerate(response["candidates"]):
        try:
            cache_value = str(jsonable_encoder(item["content"]["parts"][0]["function_call"]))
            cache_key = calculate_cache_key(cache_value)
            tool_call_id = base64.urlsafe_b64encode(
                bytes(put_cache(cache_dir, cache_key, cache_value), 'utf-8')).decode('utf-8')
            print("TEST TOOL CALL ID: ", tool_call_id)

        except KeyError:
            tool_call_id = None

        # print("FUNCTION NAME: ", item["content"]["parts"][idx]["function_call"]["name"])
        # print("FUNCTION ARGS: ", item["content"]["parts"][idx]["function_call"]["args"])

        try:
            tool_calls = [
                {
                    "id": tool_call_id,  # only on initial tool call?
                    "index": idx,
                    "type": "function",
                    "function": {
                        "name": item["content"]["parts"][idx]["function_call"]["name"],
                        "arguments": str(jsonable_encoder(item["content"]["parts"][idx]["function_call"]["args"]))
                    }
                }
            ]
        except:
            print("NO TOOL CALLS DETECTED IN RESPONSE")
            tool_calls = None

        try:
            finish_reason = map_finish_reason(item["finish_reason"])
        except KeyError:
            finish_reason = None

        try:
            content = item["content"]["parts"][0]["text"]
        except KeyError:
            content = None

        # print("TOOL CALLS: ", tool_calls)

        choice = RespChoice(
            finish_reason=finish_reason,
            index=idx,
            delta=Delta(
                role=item["content"]["role"],
                content=content,
                tool_calls=tool_calls
            ),
            logprobs=Logprobs(
                content=[
                    OAIContent(
                        token=content,
                    )
                ],
                tool_calls=tool_calls
            )
        )
        choices.append(choice)

    # print("CHOICES: ", choices)
    return OAICompletionResponse(
        id=generate_id(),
        choices=choices,
        created=0,
        model="gemini-pro",
        system_fingerprint="TEMP",
        object="chat.completion.chunk",
        # usage=Usage(
        #     completion_tokens=response.usage_metadata.candidates_token_count,
        #     prompt_tokens=response.usage_metadata.prompt_token_count,
        #     total_tokens=response.usage_metadata.total_token_count
        # )

    )


def map_finish_reason(finish_reason: str) -> str:
    print("FINISH REASON: ", finish_reason)
    # openai supports 5 stop sequences - 'stop', 'length', 'function_call', 'content_filter', 'null'
    if (finish_reason == "ERROR"):
        return "stop"
    elif (finish_reason == "FINISH_REASON_UNSPECIFIED" or finish_reason == "STOP"):
        return "stop"
    elif finish_reason == "SAFETY":  # vertex ai
        return "content_filter"
    elif finish_reason == "STOP":  # vertex ai
        return "stop"
    elif finish_reason == 1:  # vertex ai
        return "stop"
    elif finish_reason == 0:  # vertex ai
        return "stop"
    return finish_reason
