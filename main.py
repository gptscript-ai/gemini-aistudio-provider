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
from vertexai.generative_models import (Content, FunctionDeclaration, GenerationResponse, GenerativeModel, Part, Tool)

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


class Delta(Message):
    pass


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


class OAICompletionRequest(BaseModel):
    model: str
    messages: List[Message] | None = None
    max_tokens: int | None = None
    stream: bool | None = False
    seed: float | None = None
    tools: List[OAITool] | None = None
    tool_choice: List[OAITool] | Optional[str] | None = None


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


@app.get("/models")
def list_models():
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
    # print("GPTSCRIPT INPUT: ", data)
    if data.tools is not None:
        gtool_func_declarations = []
        for tool in data.tools:
            gtool_func_declaration = FunctionDeclaration(
                name=tool.function.name,
                description=tool.function.description,
                parameters=tool.function.parameters
            )

            gtool_func_declarations.append(gtool_func_declaration)
        tools = [
            Tool(gtool_func_declarations)
        ]
    else:
        tools = None

    parts = []
    for item in data.messages:
        part = Part.from_text(item.content)
        parts.append(part)
    content = Content(
        role='user',
        parts=parts
    )

    model = GenerativeModel("gemini-pro", tools=tools)
    print("CALLING GENERATE CONTENT")
    response = model.generate_content([content], tools=tools, stream=data.stream)

    return StreamingResponse(async_chunk(response), media_type="application/x-ndjson")
    # return JSONResponse(content=response_json)


async def async_chunk(chunks: Iterable[GenerationResponse]) -> AsyncIterable[str]:
    # print("CHUNKS FROM GEMINI: ")
    for chunk in chunks:
        #         print(chunk)
        chunk = map_gemini_resp(chunk)
        yield "data: " + chunk.json() + "\n\n"


def _generate_id():  # private helper function
    return "chatcmpl-" + str(uuid.uuid4())


def map_gemini_resp(response: GenerationResponse) -> OAICompletionResponse:
    choices = []
    print("PRINTING GEMINI RESPONSE AS DICT: ", response.to_dict())
    response = response.to_dict()
    for idx, item in enumerate(response["candidates"]):
        # print("ITEM: ", item)
        # tool_calls = [
        #     {
        #         "id": "some_thing",  # only on initial tool call?
        #         "index": idx,
        #         "type": "function_call",  # only on initial tool call?
        #         "function": {
        #             "name": item["content"]["parts"][0]["function_call"]["name"],
        #             "arguments": item["content"]["parts"][0]["function_call"]["args"]["question"]
        #         }
        #     }
        # ]
        # print("MAPPED TOOL CALLS: ", tool_calls)

        try:
            tool_calls = [
                {
                    "id": "some_thing",  # only on initial tool call?
                    "index": idx,
                    "type": "function",  # only on initial tool call? function_call or function?
                    "function": {
                        "name": item["content"]["parts"][0]["function_call"]["name"],
                        "arguments": item["content"]["parts"][0]["function_call"]["question"]
                    }
                }
            ]
        except:
            tool_calls = None

        try:
            finish_reason = 'continue'
            # finish_reason = map_finish_reason(item["finish_reason"])
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
                content=content
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
        id=_generate_id(),
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
    return finish_reason
