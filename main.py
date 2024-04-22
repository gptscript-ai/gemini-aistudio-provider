import json
import os
from typing import AsyncIterable, Iterable

import google.ai.generativelanguage as glm
import google.generativeai as genai
import openai.types
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRouter
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

if "GEMINI_API_KEY" in os.environ:
    api_key = os.environ["GEMINI_API_KEY"]
else:
    raise SystemExit("GEMINI_API_KEY not found in environment variables")

debug = os.environ.get("GPTSCRIPT_DEBUG", "false") == "true"
app = FastAPI()
router = APIRouter()
genai.configure(api_key=api_key)


def log(*args):
    if debug:
        print(*args)


@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()
    log("REQUEST BODY: ", body)
    return await call_next(request)


@app.get("/")
async def get_root():
    return 'ok'


async def map_tools(req_tools: list | None = None) -> list[glm.Tool] | None:
    function_declarations: list[glm.FunctionDeclaration] = []
    if req_tools:
        for tool in req_tools:
            parameters = tool['function']['parameters']
            if parameters is None:
                parameters = {
                    "properties":
                        {
                            "fake":
                                {
                                    "description": "a fake description",
                                    "type": "string"
                                }
                        },
                    "type": "object"
                }
            for item in parameters['properties']:
                parameters['properties'][item]['type_'] = parameters['properties'][item].pop('type')
                parameters['properties'][item]['type_'] = parameters['properties'][item]['type_'].upper()

            function_declarations.append(glm.FunctionDeclaration(
                name=tool['function']['name'],
                parameters={
                    "type": parameters['type'].upper(),
                    "properties": parameters['properties'],
                    "required": None,
                },
                description=tool['function']['description']
            ))
        tools: list[glm.Tool] = [glm.Tool(function_declarations=function_declarations)]
        return tools
    return None


def merge_consecutive_dicts_with_same_value(list_of_dicts, key) -> list[dict]:
    merged_list = []
    index = 0
    while index < len(list_of_dicts):
        current_dict = list_of_dicts[index]
        value_to_match = current_dict.get(key)
        compared_index = index + 1
        while compared_index < len(list_of_dicts) and list_of_dicts[compared_index].get(key) == value_to_match:
            log("CURRENT DICT: ", current_dict)
            log("COMPARED DICT: ", list_of_dicts[compared_index])
            list_of_dicts[compared_index]["content"] = current_dict["content"] + "\n" + list_of_dicts[compared_index][
                "content"]
            current_dict.update(list_of_dicts[compared_index])
            compared_index += 1
        merged_list.append(current_dict)
        index = compared_index
    return merged_list


async def map_messages(req_messages: list) -> list[glm.Content] | None:
    messages: list[glm.Content] = []
    log(req_messages)

    if req_messages is not None:
        for message in req_messages:
            match message['role']:
                case "system":
                    message['role'] = "user"
                case "user":
                    message['role'] = "user"
                case "assistant":
                    message['role'] = "model"
                case "model":
                    message['role'] = "model"
                case "tool":
                    message['role'] = "function"
                case _:
                    message['role'] = "user"
        req_messages = merge_consecutive_dicts_with_same_value(req_messages, "role")

        for message in req_messages:
            if 'tool_call_id' in message.keys():
                convert_message = glm.Content(
                    role=message['role'],
                    parts=[glm.Part(
                        function_response=glm.FunctionResponse(
                            name=message['name'],
                            response={
                                "name": message['name'],
                                "content": message['content']
                            }
                        )
                    )
                    ]
                )
            elif 'tool_calls' in message.keys():
                tool_call_parts: list[glm.Part] = []
                for tool_call in message['tool_calls']:
                    function_call = glm.FunctionCall(
                        name=tool_call['function']['name'],
                        args=json.loads(tool_call['function']['arguments'])
                    )
                    tool_call_parts.append(glm.Part(function_call=function_call))
                convert_message = glm.Content(
                    role=message['role'],
                    parts=tool_call_parts
                )
            elif 'content' in message.keys():
                convert_message = glm.Content(
                    role=message['role'],
                    parts=[glm.Part({"text": message['content']})
                           ]
                )

            messages.append(convert_message)
        return messages
    return None


@app.get("/v1/models")
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


@app.post("/v1/chat/completions")
async def chat_completion(request: Request):
    data = await request.body()
    data = json.loads(data)

    req_tools = data.get("tools", None)
    tools: list[glm.Tool] | None = None
    if req_tools is not None:
        tools = await map_tools(req_tools)

    req_messages = data["messages"]
    messages = await map_messages(req_messages)

    temperature = data.get("temperature", None)
    if temperature is not None:
        temperature = float(temperature)

    stream = data.get("stream", False)

    top_k = data.get("top_k", None)
    if top_k is not None:
        top_k = float(top_k)

    top_p = data.get("top_p", None)
    if top_p is not None:
        top_p = float(top_p)

    max_output_tokens = data.get("max_tokens", None)
    if max_output_tokens is not None:
        max_output_tokens = float(max_output_tokens)

    model = genai.GenerativeModel(data["model"])
    log("GEMINI_MESSAGES: ", messages)
    log("GEMINI_TOOLS: ", tools)
    response = model.generate_content(contents=messages,
                                      tools=tools,
                                      stream=stream,
                                      generation_config=glm.GenerationConfig(
                                          temperature=temperature,
                                          top_k=top_k,
                                          top_p=top_p,
                                          max_output_tokens=max_output_tokens,
                                      ),
                                      )

    return StreamingResponse(async_chunk(response), media_type="application/x-ndjson")


async def async_chunk(chunks: Iterable[glm.GenerateContentResponse]) -> \
        AsyncIterable[str]:
    for chunk in chunks:
        mapped_chunk = map_streaming_resp(chunk)
        log("MAPPED CHUNK: ", mapped_chunk.model_dump_json())
        yield "data: " + mapped_chunk.model_dump_json() + "\n\n"


def map_streaming_resp(
        chunk: glm.GenerateContentResponse) -> ChatCompletionChunk | None:
    tool_calls = []
    for outer, candidate in enumerate(chunk.candidates):
        content: str | None = None
        for inner, part in enumerate(candidate.content.parts):
            tool_call_id = part.function_call.name
            if tool_call_id is not None and tool_call_id != "":
                args = dict(part.function_call.args)
                for key, value in args.items():
                    if isinstance(args[key], str):
                        args[key] = value.replace('\\', '')

                tool_calls.append({
                    "id": tool_call_id,
                    "index": inner,
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": json.dumps(args),
                    }
                })

            try:
                content = part.text
            except KeyError:
                content = None

        role: str
        match chunk.candidates[0].content.role:
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
            if len(tool_calls) > 0:
                finish_reason = "tool_calls"
            else:
                finish_reason = map_finish_reason(str(candidate.finish_reason))
        except KeyError:
            finish_reason = None

        log("FINISH_REASON: ", finish_reason)

        resp = ChatCompletionChunk(
            id="0",
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=content,
                        tool_calls=tool_calls,
                        role=role
                    ),
                    finish_reason=finish_reason,
                    index=0,
                )
            ],
            created=0,
            model="",
            object="chat.completion.chunk",
        )
        return resp
    return None


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
    elif finish_reason == '1':
        return "stop"
    elif finish_reason == '0':
        return "stop"
    return finish_reason


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")),
                log_level="debug" if debug else "critical", reload=debug, access_log=debug)
