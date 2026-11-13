"""
FastAPI приложение для генерации карточек товаров.

Сервис принимает данные о контрольно-измерительном приборе и генерирует
структурированную карточку для B2B-магазина СОЮЗ-ПРИБОР с помощью LLM.

Тестовое задание для компании СОЮЗ-ПРИБОР.
Автор: Бахимов Олег Николаевич
Дата: 13 ноября 2026 г.
"""

import os
import logging
from datetime import datetime
from typing import List
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ===== 1. ЯВНАЯ ЗАГРУЗКА .ENV ДО ВСЕХ ОСТАЛЬНЫХ ИМПОРТОВ =====
# КРИТИЧНО: load_dotenv должен быть ПЕРЕД импортом llm_client,
# иначе LLMClient инициализируется со значениями по умолчанию
current_dir = Path(__file__).parent
env_path = current_dir / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"[DEBUG] .env загружен из: {env_path}")
else:
    print(f"[WARNING] Файл .env не найден по пути: {env_path}")

# Диагностический вывод значений из .env
print(f"[DEBUG] LLM_PROVIDER = {os.getenv('LLM_PROVIDER', 'NOT_SET')}")
print(f"[DEBUG] LLM_MODEL    = {os.getenv('LLM_MODEL', 'NOT_SET')}")
print(f"[DEBUG] OPENAI_BASE_URL = {os.getenv('OPENAI_BASE_URL', 'NOT_SET')}")
print(f"[DEBUG] ROUTERAI_API_KEY = {'SET' if os.getenv('ROUTERAI_API_KEY') else 'NOT_SET'}")


# ===== 2. ИМПОРТЫ ПРОЕКТА (после load_dotenv!) =====
from models import (
    ProductInput,
    ProductOutput,
    SEOContent,
    ProductSpec,
    ErrorResponse,
)
from prompts import SYSTEM_PROMPT, GENERATION_PROMPT, format_product_data
from llm_client import LLMClient


# ===== 3. НАСТРОЙКА ЛОГИРОВАНИЯ =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===== 4. СОЗДАНИЕ FASTAPI ПРИЛОЖЕНИЯ =====
app = FastAPI(
    title="AI Product Description Generator",
    description=(
        "Сервис генерации карточек товаров для B2B-магазина СОЮЗ-ПРИБОР. "
        "Использует LLM (RouterAI/OpenAI/Anthropic) для создания SEO-оптимизированных "
        "описаний контрольно-измерительных приборов."
    ),
    version="1.0.0",
)


# ===== 5. CORS (для доступа из Swagger UI и фронтенда) =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 6. ИНИЦИАЛИЗАЦИЯ LLM КЛИЕНТА =====
llm_client: LLMClient | None = None

try:
    llm_client = LLMClient()
    logger.info(f"LLM клиент успешно инициализирован: {llm_client.provider} / {llm_client.model}")
except Exception as e:
    logger.error(f"Ошибка инициализации LLM клиента: {e}")
    # Не прерываем запуск — даём возможность диагностировать через /health


# ===== 7. ENDPOINTS =====

@app.get("/")
async def root() -> dict:
    """Корневой endpoint с информацией о сервисе"""
    return {
        "service": "AI Product Description Generator",
        "version": "1.0.0",
        "author": "Бахимов О.Н.",
        "task": "Тестовое задание СОЮЗ-ПРИБОР",
        "endpoints": {
            "GET /health": "Проверка работоспособности",
            "GET /debug/model": "Диагностика LLM-модели",
            "POST /generate": "Генерация карточки товара",
            "POST /generate-batch": "Пакетная генерация",
            "GET /docs": "Swagger UI документация",
            "GET /redoc": "ReDoc документация",
        },
    }


@app.get("/health")
async def health_check() -> dict:
    """Проверка работоспособности сервиса"""
    if llm_client is None:
        return {
            "status": "error",
            "message": "LLM клиент не инициализирован",
            "timestamp": datetime.now().isoformat(),
        }
    
    return {
        "status": "ok",
        "provider": llm_client.provider,
        "model": llm_client.model,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/debug/model")
async def debug_model() -> dict:
    """
    Диагностика LLM-модели.
    
    Проверяет доступность модели через API и возвращает
    список доступных моделей для диагностики.
    """
    # Проверка 1: LLM клиент инициализирован
    if llm_client is None:
        return {"error": "LLM клиент не инициализирован. Проверьте .env"}
    
    # Проверка 2: используем OpenAI-совместимый провайдер
    if llm_client.provider != "openai":
        return {
            "error": "Эндпоинт работает только для OpenAI-совместимых провайдеров",
            "current_provider": llm_client.provider,
        }
    
    # Проверка 3: OpenAI клиент существует
    if llm_client.openai_client is None:
        return {"error": "OpenAI клиент не инициализирован внутри LLMClient"}
    
    # Сохраняем в локальную переменную для type narrowing
    client = llm_client.openai_client
    target_model = llm_client.model
    
    try:
        # Получаем список доступных моделей
        models_response = client.models.list()
        model_ids = [m.id for m in models_response.data]
        
        is_available = target_model in model_ids
        
        # Тестовый вызов с инициализацией переменных по умолчанию
        test_error: str | None = None
        test_success = False
        test_tokens: int = 0  # Явная инициализация
        
        try:
            test_response = client.chat.completions.create(
                model=target_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            test_success = True
            if test_response.usage:
                test_tokens = test_response.usage.total_tokens
        except Exception as e:
            test_error = str(e)
        
        return {
            "target_model": target_model,
            "is_available": is_available,
            "test_call_success": test_success,
            "test_tokens_used": test_tokens if test_success else None,
            "test_error": test_error,
            "available_models_count": len(model_ids),
            "similar_models": [
                m for m in model_ids
                if "gpt" in m.lower() or "terra" in m.lower() or "6.1" in m
            ][:15],
            "provider_url": os.getenv("OPENAI_BASE_URL"),
        }
    except Exception as e:
        logger.error(f"Ошибка в /debug/model: {e}", exc_info=True)
        return {"error": str(e)}

@app.post("/generate", response_model=ProductOutput)
async def generate_product_card(product: ProductInput) -> ProductOutput:
    """
    Генерирует карточку товара по входным данным.
    
    Принимает JSON с информацией о товаре и возвращает
    структурированную карточку с SEO-контентом.
    """
    if llm_client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM клиент не инициализирован. Проверьте .env файл и API-ключ.",
        )
    
    try:
        # Форматируем данные для промпта
        formatted_data = format_product_data(product.model_dump())
        user_prompt = GENERATION_PROMPT.format(**formatted_data)
        
        logger.info(f"Генерация для товара: {product.name}")
        
        # Генерируем через LLM
        result = llm_client.generate(SYSTEM_PROMPT, user_prompt)
        
        if not result["success"]:
            logger.error(f"Ошибка генерации: {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка генерации: {result.get('error')}",
            )
        
        content_data = result["content"]
        
        # Безопасное преобразование structured_specs в ProductSpec объекты
        structured_specs: List[ProductSpec] = []
        raw_specs = content_data.get("structured_specs", [])
        if isinstance(raw_specs, list):
            for spec in raw_specs:
                if isinstance(spec, dict) and "name" in spec and "value" in spec:
                    structured_specs.append(
                        ProductSpec(name=str(spec["name"]), value=str(spec["value"]))
                    )
        
        # Формируем ответ
        return ProductOutput(
            product_name=product.name,
            manufacturer=product.manufacturer,
            content=SEOContent(
                seo_title=str(content_data.get("seo_title", "")),
                seo_description=str(content_data.get("seo_description", "")),
                structured_specs=structured_specs,
                search_queries=list(content_data.get("search_queries", [])),
                commercial_block=str(content_data.get("commercial_block", "")),
                missing_data=list(content_data.get("missing_data", [])),
            ),
            llm_provider=result["provider"],
            llm_model=result["model"],
            processing_time_seconds=result["processing_time"],
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка при генерации: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")


@app.post("/generate-batch")
async def generate_batch(products: List[ProductInput]) -> dict:
    """
    Генерирует карточки для нескольких товаров (batch-обработка).
    
    Возвращает aggregated результат: сколько успешно, сколько с ошибками.
    """
    if llm_client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM клиент не инициализирован",
        )
    
    results: List[ProductOutput] = []
    errors: List[dict] = []
    
    for i, product in enumerate(products):
        try:
            result = await generate_product_card(product)
            results.append(result)
            logger.info(f"[{i + 1}/{len(products)}] Успешно: {product.name}")
        except HTTPException as e:
            errors.append({
                "product_name": product.name,
                "error": e.detail,
                "status_code": e.status_code,
            })
            logger.error(f"[{i + 1}/{len(products)}] Ошибка: {product.name} - {e.detail}")
        except Exception as e:
            errors.append({
                "product_name": product.name,
                "error": str(e),
                "status_code": 500,
            })
            logger.error(f"[{i + 1}/{len(products)}] Неожиданная ошибка: {product.name} - {e}")
    
    return {
        "total": len(products),
        "success": len(results),
        "errors": len(errors),
        "results": [r.model_dump() for r in results],
        "error_details": errors,
    }


# ===== 8. ТОЧКА ВХОДА =====
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("  AI Product Description Generator")
    print("  Swagger UI: http://localhost:8000/docs")
    print("  ReDoc:      http://localhost:8000/redoc")
    print("  Health:     http://localhost:8000/health")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )