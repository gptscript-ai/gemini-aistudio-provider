import json
import os
import uuid
from typing import AsyncIterable, Iterable, List, Optional

import google.ai.generativelanguage as glm
import google.generativeai as genai
import openai.types
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel

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
    pass


class OAIParameters(BaseModel):
    type: str
    properties: dict | None = None
    required: list[str] | None = []


class OAIFunction(BaseModel):
    name: str
    description: str
    parameters: OAIParameters


class OAITool(BaseModel):
    type: str | str = "function"
    function: OAIFunction


class OAIToolCall(OAITool):
    id: str | None = None

    def generate_id(self) -> str:
        return self.id if self.id else generate_id("call")


def map_oai_to_gemini_function(oai_function: OAIFunction) -> glm.FunctionDeclaration:
    for item in oai_function.parameters.properties:
        oai_function.parameters.properties[item]["type_"] = oai_function.parameters.properties[item].pop("type")
        oai_function.parameters.properties[item]["type_"] = oai_function.parameters.properties[item][
            "type_"].upper()
        oai_function.parameters.properties[item].pop("description")

    return glm.FunctionDeclaration(
        name=oai_function.name,
        description=oai_function.description,
        parameters={
            "type": oai_function.parameters.type.upper(),
            "properties": oai_function.parameters.properties,
            "required": oai_function.parameters.required,
        }
    )


# Maps OpenAI message objects to Gemini content objects
def map_oai_to_gemini_content(oai_messages: List[Message]) -> list[glm.Content]:
    user_parts: list[glm.Part] = []
    model_parts: list[glm.Part] = []
    function_parts: list[glm.Part] = []
    for message in oai_messages:
        if message.tool_call_id:
            part = glm.Part(function_response=glm.FunctionResponse(
                name=message.tool_call_id,
                response={
                    "name": message.tool_call_id,
                    "content": message.content
                }
            ))
        elif message.tool_calls:
            for tool_call in message.tool_calls:
                part = glm.Part(
                    function_call=glm.FunctionCall(
                        name=tool_call["function"]["name"],
                        args=json.loads(tool_call["function"]["arguments"])
                    )
                )
        elif message.content:
            part = glm.Part({"text": message.content})

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

    gemini_content: list[glm.Content] = []

    if user_parts:
        gemini_content.append(glm.Content({
            "role": "user",
            "parts": user_parts
        }))

    if model_parts:
        gemini_content.append(glm.Content({
            "role": "model",
            "parts": model_parts
        }))

    if function_parts:
        gemini_content.append(glm.Content({
            "role": "function",
            "parts": function_parts
        }))

    return gemini_content


def oai_to_gemini_tools(oai_tools: list[OAITool]) -> list[glm.Tool]:
    if oai_tools is not None:
        gemini_functions: list[glm.FunctionDeclaration] = []
        for tool in oai_tools:
            gemini_function = map_oai_to_gemini_function(tool.function)
            gemini_functions.append(gemini_function)

        return [glm.Tool({
            "function_declarations": gemini_functions
        })]


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


class RespChoice(BaseModel):
    finish_reason: str | None = None
    index: int
    delta: Delta
    logprobs: Logprobs
    tool_calls: List[OAIToolCall] | None = None


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
    top_k: int | None = None
    top_p: float | None = None


@app.get("/models")
def list_models() -> JSONResponse:
    models = []
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            model = openai.types.Model(id=model.name.removeprefix("models/"), created=0, object="model",
                                       owned_by="system")
            models.append(model)

    models = {"data": models}
    models_json = jsonable_encoder(models)
    return JSONResponse(content=models_json)


@app.post("/chat/completions", response_model_exclude_none=True, response_model_exclude_unset=True,
          response_model=OAICompletionResponse)
async def chat_completion(data: OAICompletionRequest):
    if data.tools is not None:
        gemini_tools = oai_to_gemini_tools(data.tools)
    else:
        gemini_tools = None

    messages: list[glm.Content] = map_oai_to_gemini_content(data.messages)

    model = genai.GenerativeModel(data.model)
    print("GEMINI_MESSAGES: ", messages)
    print("GEMINI_TOOLS: ", gemini_tools)
    response = model.generate_content(messages, tools=gemini_tools, stream=data.stream,

                                      generation_config=genai.generative_models.generation_types.GenerationConfig(
                                          temperature=data.temperature,
                                          top_k=data.top_k,
                                          top_p=data.top_p,
                                          max_output_tokens=data.max_tokens,
                                      ),
                                      )

    return StreamingResponse(async_chunk(response), media_type="application/x-ndjson")


async def async_chunk(chunks: Iterable[genai.generative_models.generation_types.GenerateContentResponse]) -> \
        AsyncIterable[str]:
    for chunk in chunks:
        chunk = map_gemini_to_oai_response(chunk)
        print("MAPPED CHUNK: ", chunk.json())
        yield "data: " + chunk.json() + "\n\n"


def generate_id(type: str = "chatcmpl") -> str:
    return type + "-" + str(uuid.uuid4())


def map_gemini_to_oai_response(
        response: genai.generative_models.generation_types.GenerateContentResponse) -> OAICompletionResponse:
    choices = []
    tool_calls = []
    # response = dict_from_class(response)
    # print("GEMINI RESPONSE: ", dict_from_class(response.parts))
    print("UNTOUCHED_RESPONSE: ", response)

    # print("GENERATION RESPONSE: ", response.candidates[0].content.role)

    for outer, candidate in enumerate(response.candidates):
        content: str | None = None
        for inner, part in enumerate(response.parts):

            tool_call_id = part.function_call.name.replace("_", "-")
            if tool_call_id is not None and tool_call_id != "":
                tool_calls.append({
                    "id": tool_call_id,
                    "index": inner,
                    "type": "function",
                    "function": {
                        "name": part.function_call.name.replace("_", "-"),
                        "arguments": json.dumps(dict(part.function_call.args)),
                    }
                })
            else:
                tool_calls = None

            try:
                content = part.text
            except KeyError:
                content = None

        role: str
        match candidate.content.role:
            case "system":
                role = "user"
            case "user":
                role = "user"
            case "assistant":
                role = "model"
            case "model":
                role = "assistant"
            case "function":
                role = "tool"
            case _:
                role = "user"

        try:
            finish_reason = map_finish_reason(candidate.finish_reason)
        except KeyError:
            finish_reason = None

        choice = RespChoice(
            finish_reason=finish_reason,
            index=outer,
            delta=Delta(
                role=role,
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

    return OAICompletionResponse(
        id=generate_id(),
        choices=choices,
        created=0,
        model="gemini-pro",
        system_fingerprint="TEMP",
        object="chat.completion.chunk",
    )


def map_finish_reason(finish_reason: str) -> str:
    # openai supports 5 stop sequences - 'stop', 'length', 'function_call', 'content_filter', 'null'
    if (finish_reason == "ERROR"):
        return "stop"
    elif (finish_reason == "FINISH_REASON_UNSPECIFIED" or finish_reason == "STOP"):
        return "stop"
    elif finish_reason == "SAFETY":
        return "content_filter"
    elif finish_reason == "STOP":
        return "stop"
    elif finish_reason == 1:
        return "stop"
    elif finish_reason == 0:
        return "stop"
    return finish_reason
