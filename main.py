import base64
import json
import os
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
    pass


class OAIParameters(BaseModel):
    type: str
    properties: dict | None = None
    required: list[str] | None = None


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


class GeminiParameters(BaseModel):
    type: str
    properties: dict | None = None
    required: list[str] | None = None


class GeminiFunction(BaseModel):
    name: str
    description: str
    parameters: GeminiParameters

    def convert_to_sdk_function_declaration(self) -> FunctionDeclaration:
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters.dict()
        )


class GeminiTool(BaseModel):
    function_declarations: list[GeminiFunction] | None = None

    def convert_to_sdk_tool(self) -> Tool:
        functions: list[FunctionDeclaration] = []
        for function in self.function_declarations:
            functions.append(function.convert_to_sdk_function_declaration())
        return Tool(functions)


def map_oai_to_gemini_function(oai_function: OAIFunction) -> GeminiFunction:
    return GeminiFunction(
        name=oai_function.name,
        description=oai_function.description,
        parameters=GeminiParameters(
            type=oai_function.parameters.type,
            properties=oai_function.parameters.properties
        )
    )


def map_gemini_to_oai_function(gemini_function: GeminiFunction) -> OAIFunction:
    oai_function = OAIFunction()
    oai_function.name = gemini_function.name
    oai_function.description = gemini_function.description
    oai_function.parameters = gemini_function.parameters
    return oai_function


# Maps OpenAI message objects to Gemini content objects
def map_oai_to_gemini_content(oai_messages: List[Message]) -> list[Content]:
    user_parts: list[Part] = []
    model_parts: list[Part] = []
    function_parts: list[Part] = []
    for message in oai_messages:
        if message.tool_call_id:
            decoded_tool_call_id = base64.urlsafe_b64decode(bytes(message.tool_call_id, 'utf-8')).decode('utf-8')
            part = Part.from_function_response(name=decoded_tool_call_id,
                                               response={
                                                   "name": decoded_tool_call_id,
                                                   "content": message.content
                                               })
        elif message.tool_calls:
            for tool_call in message.tool_calls:
                part = Part.from_dict({
                    "function_call": {
                        "name": tool_call["function"]["name"]
                    }
                })
        elif message.content:
            part = Part.from_text(message.content)

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

    gemini_content: list[Content] = []

    if user_parts:
        gemini_content.append(Content(
            role="user",
            parts=user_parts
        ))

    if model_parts:
        gemini_content.append(Content(
            role="model",
            parts=model_parts
        ))

    if function_parts:
        gemini_content.append(Content(
            role="function",
            parts=function_parts
        ))

    return gemini_content


def oai_to_gemini_tools(oai_tools: list[OAITool]) -> list[Tool]:
    gemini_tools: list[Tool] | None = None
    if oai_tools is not None:
        gemini_functions: list[GeminiFunction] = []
        for tool in oai_tools:
            gemini_function = map_oai_to_gemini_function(tool.function)
            gemini_functions.append(gemini_function)
        return [GeminiTool(function_declarations=gemini_functions).convert_to_sdk_tool()]


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
        model = openai.types.Model(id=model.name.removeprefix("models/"), created=0, object="model", owned_by="system")
        models.append(model)

    models = {"data": models}
    models_json = jsonable_encoder(models)
    return JSONResponse(content=models_json)


@app.post("/chat/completions", response_model_exclude_none=True, response_model_exclude_unset=True,
          response_model=OAICompletionResponse)
async def chat_completion(data: OAICompletionRequest):
    gemini_tools = oai_to_gemini_tools(data.tools)
    messages: list[Content] = map_oai_to_gemini_content(data.messages)

    model = GenerativeModel("gemini-pro", tools=gemini_tools)
    response = model.generate_content(messages, tools=gemini_tools, stream=data.stream,
                                      generation_config=GenerationConfig(
                                          temperature=data.temperature,
                                          top_k=data.top_k,
                                          top_p=data.top_p,
                                          max_output_tokens=data.max_tokens,
                                      ))

    return StreamingResponse(async_chunk(response), media_type="application/x-ndjson")


async def async_chunk(chunks: Iterable[GenerationResponse]) -> AsyncIterable[str]:
    for chunk in chunks:
        chunk = map_gemini_to_oai_response(chunk)
        yield "data: " + chunk.json() + "\n\n"


def generate_id(type: str = "chatcmpl") -> str:
    return type + "-" + str(uuid.uuid4())


def map_gemini_to_oai_response(response: GenerationResponse) -> OAICompletionResponse:
    choices = []
    response = response.to_dict()
    for idx, item in enumerate(response["candidates"]):
        try:
            tool_call_id = base64.urlsafe_b64encode(
                bytes(item["content"]["parts"][0]["function_call"]["name"], 'utf-8')).decode(
                'utf-8')
        except:
            tool_call_id = None

        try:
            tool_calls = [
                {
                    "id": tool_call_id,
                    "index": idx,
                    "type": "function",
                    "function": {
                        "name": item["content"]["parts"][0]["function_call"]["name"],
                        "arguments": json.dumps(item["content"]["parts"][0]["function_call"]["args"])
                    }
                }
            ]
        except:
            tool_calls = None

        try:
            finish_reason = map_finish_reason(item["finish_reason"])
        except KeyError:
            finish_reason = None

        try:
            content = item["content"]["parts"][0]["text"]
        except KeyError:
            content = None

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
