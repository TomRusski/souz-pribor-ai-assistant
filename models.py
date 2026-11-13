"""Pydantic модели для валидации данных"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class ProductInput(BaseModel):
    """Входные данные о товаре"""
    name: str = Field(..., description="Название товара", min_length=3)
    manufacturer: str = Field(..., description="Производитель")
    category: Optional[str] = Field(None, description="Категория")
    price: Optional[str] = Field(None, description="Цена")
    currency: Optional[str] = Field("RUR", description="Валюта")
    specs: Dict[str, str] = Field(default_factory=dict, description="Характеристики")
    description: Optional[str] = Field("", description="Описание")
    applications: Optional[List[str]] = Field(default_factory=list)
    features: Optional[List[str]] = Field(default_factory=list)
    url: Optional[str] = Field(None, description="URL товара")


class ProductSpec(BaseModel):
    """Техническая характеристика"""
    name: str
    value: str


class SEOContent(BaseModel):
    """Сгенерированный контент"""
    seo_title: str = Field(..., max_length=70)
    seo_description: str
    structured_specs: List[ProductSpec]
    search_queries: List[str]
    commercial_block: str
    missing_data: List[str] = Field(default_factory=list)


class ProductOutput(BaseModel):
    """Полный результат"""
    product_name: str
    manufacturer: str
    generated_at: datetime = Field(default_factory=datetime.now)
    content: SEOContent
    llm_provider: str
    llm_model: str
    processing_time_seconds: float


class ErrorResponse(BaseModel):
    """Ответ об ошибке"""
    error: str
    details: Optional[str] = None
    product_name: Optional[str] = None