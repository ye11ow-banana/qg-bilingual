from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, get_type_hints


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


try:  # pragma: no cover - prefer real Pydantic when available
    import pydantic  # type: ignore
except ImportError:  # pragma: no cover - lightweight stub for offline testing
    class BaseModel:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, data: Dict[str, Any]) -> "BaseModel":
            return cls(**data)

        def model_dump(self) -> Dict[str, Any]:
            return self.__dict__

    def Field(default: Any = None, description: str | None = None):  # type: ignore
        return default

    pydantic_module = types.ModuleType("pydantic")
    pydantic_module.BaseModel = BaseModel
    pydantic_module.Field = Field
    sys.modules["pydantic"] = pydantic_module


try:  # pragma: no cover - prefer real FastAPI when available
    import fastapi  # type: ignore
except ImportError:  # pragma: no cover - lightweight stub for offline testing
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: Any) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    @dataclass
    class Depends:
        dependency: Callable[..., Any]

    class FastAPI:
        def __init__(self, title: str | None = None) -> None:  # noqa: D401 - stub
            self.title = title
            self._routes: Dict[Tuple[str, str], Callable[..., Any]] = {}

        def get(self, path: str, response_model: Any | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return self._register("GET", path)

        def post(self, path: str, response_model: Any | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return self._register("POST", path)

        def _register(self, method: str, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                self._routes[(method.upper(), path)] = func
                return func

            return decorator

    class _Response:
        def __init__(self, status_code: int, payload: Any) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> Any:
            return self._payload

    class TestClient:
        def __init__(self, app: FastAPI) -> None:
            self.app = app

        def post(self, path: str, json: Dict[str, Any]) -> _Response:
            return self._call("POST", path, json)

        def get(self, path: str) -> _Response:
            return self._call("GET", path, {})

        def _call(self, method: str, path: str, json: Dict[str, Any]) -> _Response:
            handler = self.app._routes.get((method, path))
            if handler is None:
                return _Response(404, {"detail": "Not Found"})

            try:
                bound_args = self._prepare_args(handler, json)
                result = handler(**bound_args)
                status = 200
                payload = result
                if hasattr(result, "model_dump"):
                    payload = result.model_dump()
                return _Response(status, payload)
            except HTTPException as exc:  # type: ignore[no-untyped-call]
                return _Response(exc.status_code, {"detail": exc.detail})

        def _prepare_args(self, handler: Callable[..., Any], json_payload: Dict[str, Any]) -> Dict[str, Any]:
            sig = signature(handler)
            hints = get_type_hints(handler)
            bound: Dict[str, Any] = {}
            for name, param in sig.parameters.items():
                hint = hints.get(name, param.annotation)
                if isinstance(param.default, Depends):
                    bound[name] = param.default.dependency()
                elif hint and hasattr(hint, "model_validate"):
                    bound[name] = hint.model_validate(json_payload)
                else:
                    bound[name] = json_payload.get(name)
            return bound

    TestClient.__test__ = False  # prevent pytest from collecting the stub

    fastapi_module = types.ModuleType("fastapi")
    fastapi_module.FastAPI = FastAPI
    fastapi_module.Depends = Depends
    fastapi_module.HTTPException = HTTPException
    testclient_module = types.ModuleType("fastapi.testclient")
    testclient_module.TestClient = TestClient
    sys.modules["fastapi"] = fastapi_module
    sys.modules["fastapi.testclient"] = testclient_module
