"""Клиент для работы с LLM API через RouterAI.ru / OpenAI / Anthropic"""

import os
import json
import time
import logging
import re
from typing import Dict, Any, Optional
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from anthropic import Anthropic
from anthropic.types.message import Message

logger = logging.getLogger(__name__)


class LLMClient:
    """Универсальный клиент (RouterAI / OpenAI / Anthropic)"""
    
    def __init__(self) -> None:
        self.provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model: str = os.getenv("LLM_MODEL", "openai/gpt-6.1-terra-pro")
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
        self.max_retries: int = 3
        self.retry_delay: int = 2
        
        # Раздельные клиенты для каждого провайдера
        self.openai_client: Optional[OpenAI] = None
        self.anthropic_client: Optional[Anthropic] = None
        
        if self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Неподдерживаемый провайдер: {self.provider}")
    
    def _init_anthropic(self) -> None:
        """Инициализация Anthropic клиента"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY не установлен в .env")
        self.anthropic_client = Anthropic(api_key=api_key)
        logger.info(f"Anthropic клиент: {self.model}")
    
    def _init_openai(self) -> None:
        """
        Инициализация OpenAI-совместимого клиента.
        Работает с RouterAI, OpenAI, OpenRouter, Ollama.
        """
        api_key = os.getenv("ROUTERAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Не установлен API-ключ. Добавьте в .env: "
                "ROUTERAI_API_KEY (для routerai.ru) или OPENAI_API_KEY"
            )
        
        base_url = os.getenv("OPENAI_BASE_URL", "https://routerai.ru/api/v1")
        timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
        
        self.openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        
        logger.info(
            f"OpenAI-совместимый клиент: {self.model} через {base_url}"
        )
    
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Генерирует ответ от LLM с retry-логикой"""
        start_time = time.time()
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                if self.provider == "anthropic":
                    response = self._call_anthropic(system_prompt, user_prompt)
                else:
                    response = self._call_openai(system_prompt, user_prompt)
                
                processing_time = time.time() - start_time
                parsed_content = self._parse_json_response(response)
                
                return {
                    "content": parsed_content,
                    "raw_response": response,
                    "processing_time": processing_time,
                    "provider": self.provider,
                    "model": self.model,
                    "success": True
                }
                
            except Exception as e:
                last_error = e
                
                if self._is_non_retryable_error(e):
                    logger.error(f"Критическая ошибка (без retry): {e}")
                    break
                
                logger.warning(f"Попытка {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        processing_time = time.time() - start_time
        return {
            "content": None,
            "error": str(last_error),
            "processing_time": processing_time,
            "provider": self.provider,
            "model": self.model,
            "success": False
        }
    
    def _is_non_retryable_error(self, error: Exception) -> bool:
        """Проверяет, что ошибка не требует повторных попыток"""
        error_str = str(error).lower()
        non_retryable_keywords = [
            "invalid_api_key",
            "authentication",
            "does not exist",
            "model_not_found",
            "invalid_request_error",
            "billing",
            "quota exceeded",
            "insufficient_quota",
            "insufficient balance",
            "недостаточно средств"
        ]
        return any(kw in error_str for kw in non_retryable_keywords)
    
    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Вызов Anthropic API"""
        if self.anthropic_client is None:
            raise RuntimeError("Anthropic клиент не инициализирован")
        
        response: Message = self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        # Anthropic возвращает список блоков (TextBlock, ToolUseBlock и др.)
        # Нам нужен только текстовый блок
        for block in response.content:
            # Проверяем тип блока через getattr для безопасности
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_content = getattr(block, "text", None)
                if text_content is not None:
                    return str(text_content)
        
        raise ValueError("Не удалось извлечь текст из ответа Anthropic")
    
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Вызов OpenAI-совместимого API (RouterAI/OpenAI/OpenRouter)"""
        if self.openai_client is None:
            raise RuntimeError("OpenAI клиент не инициализирован")
        
        response: ChatCompletion = self.openai_client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Проверяем, что content не None
        content: Optional[str] = response.choices[0].message.content
        if content is None:
            raise ValueError("Пустой ответ от LLM (content is None)")
        
        # Проверяем finish_reason
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            logger.warning(
                f"Ответ обрезан по max_tokens ({self.max_tokens}). "
                f"Увеличьте MAX_TOKENS в .env"
            )
        
        # Логируем использование токенов
        if response.usage:
            logger.info(
                f"Токены: prompt={response.usage.prompt_tokens}, "
                f"completion={response.usage.completion_tokens}, "
                f"total={response.usage.total_tokens}"
            )
        
        return content
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Парсит JSON из ответа LLM"""
        if response is None:
            raise ValueError("Пустой ответ от LLM")
        
        response = response.strip()
        
        # Удаляем markdown-обёртки
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            raise ValueError(
                f"Не удалось распарсить JSON. Первые 200 символов: {response[:200]}"
            )