# Capítulo 08 — Deploy e Integração: Levando RAG para Produção

Depois de construir e otimizar nosso sistema RAG, é hora de colocá-lo em produção. Este capítulo aborda como criar uma API robusta com FastAPI, containerizar a aplicação com Docker e integrá-la com sistemas externos.

Transformaremos nosso protótipo local em um serviço escalável, acessível via REST API, pronto para integração com aplicações web, móveis e outros sistemas.

**Resumo:** neste capítulo criamos uma API REST com FastAPI para servir nosso sistema RAG, containerizamos com Docker e implementamos estratégias de integração com aplicações externas.

**Sumário:**
1. Arquitetura de deploy para RAG
2. Criando API REST com FastAPI
3. Validação e tratamento de erros
4. Autenticação e segurança
5. Containerização com Docker
6. Configuração e variáveis de ambiente
7. Integração com aplicações externas
8. Monitoramento e logs
9. Estratégias de deployment
10. Conclusão

---

## 1. Arquitetura de deploy para RAG

### 1.1. Componentes da arquitetura

Um sistema RAG em produção típico inclui:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │   RAG Service   │
│   (Web/Mobile)  │◄──►│   (FastAPI)     │◄──►│   (Core Logic)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▲                        ▲
                                │                        │
                                ▼                        ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   Auth Service  │    │   Vector DB     │
                    │   (JWT/OAuth)   │    │   (ChromaDB)    │
                    └─────────────────┘    └─────────────────┘
```

### 1.2. Requisitos não-funcionais

```python
# requirements_production.txt
# Core RAG
langchain>=0.1.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
openai>=1.0.0

# API Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0

# Security & Auth
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6

# Monitoring & Logging
prometheus-client>=0.19.0
structlog>=23.2.0
sentry-sdk[fastapi]>=1.38.0

# Production
gunicorn>=21.2.0
redis>=5.0.0
sqlalchemy>=2.0.0
alembic>=1.12.0
```

---

## 2. Criando API REST com FastAPI

### 2.1. Estrutura básica da API

```python
# api/main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import uuid
from datetime import datetime

from rag_demo.rag import RAGSystem
from rag_demo.config import settings

# Inicializar FastAPI
app = FastAPI(
    title="RAG API",
    description="Sistema RAG para consultas inteligentes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global RAG instance
rag_system = RAGSystem()
```

### 2.2. Modelos de dados (Pydantic)

```python
# api/models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class QueryRequest(BaseModel):
    query: str = Field(
        ..., 
        min_length=3, 
        max_length=1000,
        description="Pergunta a ser respondida"
    )
    max_results: Optional[int] = Field(
        default=5, 
        ge=1, 
        le=20,
        description="Número máximo de documentos a recuperar"
    )
    threshold: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Limiar de similaridade"
    )
    use_guardrails: Optional[bool] = Field(
        default=True,
        description="Aplicar guardrails na resposta"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="ID da conversa para contexto"
    )

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query não pode estar vazia')
        return v.strip()

class Source(BaseModel):
    content: str = Field(..., description="Trecho do documento")
    similarity: float = Field(..., description="Pontuação de similaridade")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryResponse(BaseModel):
    request_id: str = Field(..., description="ID único da requisição")
    query: str = Field(..., description="Pergunta original")
    answer: str = Field(..., description="Resposta gerada")
    sources: List[Source] = Field(default_factory=list)
    confidence: float = Field(..., description="Confiança na resposta")
    processing_time: float = Field(..., description="Tempo de processamento")
    timestamp: datetime = Field(default_factory=datetime.now)
    guardrails_triggered: List[str] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str = Field(..., description="Status do serviço")
    version: str = Field(..., description="Versão da API")
    timestamp: datetime = Field(default_factory=datetime.now)
    database_status: str = Field(..., description="Status do banco vetorial")
    model_status: str = Field(..., description="Status do modelo")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Tipo do erro")
    message: str = Field(..., description="Mensagem de erro")
    request_id: Optional[str] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)
```

### 2.3. Endpoints principais

```python
# api/endpoints.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPAuthorizationCredentials
import time
import uuid
import structlog

from .models import QueryRequest, QueryResponse, HealthResponse, ErrorResponse
from .auth import verify_token
from .metrics import record_query_metrics

logger = structlog.get_logger()

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Endpoint principal para consultas RAG"""
    
    # Verificar autenticação
    user_id = await verify_token(credentials.credentials)
    
    # Gerar ID da requisição
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(
            "query_received",
            request_id=request_id,
            user_id=user_id,
            query=request.query[:100]  # Log apenas primeiros 100 chars
        )
        
        # Executar consulta RAG
        result = await rag_system.aquery(
            query=request.query,
            max_results=request.max_results,
            threshold=request.threshold,
            use_guardrails=request.use_guardrails,
            conversation_id=request.conversation_id
        )
        
        processing_time = time.time() - start_time
        
        # Preparar resposta
        response = QueryResponse(
            request_id=request_id,
            query=request.query,
            answer=result["answer"],
            sources=[
                Source(
                    content=doc.page_content,
                    similarity=doc.metadata.get("similarity", 0.0),
                    metadata=doc.metadata
                )
                for doc in result["source_documents"]
            ],
            confidence=result.get("confidence", 0.0),
            processing_time=processing_time,
            guardrails_triggered=result.get("guardrails_triggered", [])
        )
        
        # Metrics em background
        background_tasks.add_task(
            record_query_metrics,
            request_id=request_id,
            user_id=user_id,
            processing_time=processing_time,
            confidence=response.confidence,
            num_sources=len(response.sources)
        )
        
        logger.info(
            "query_completed",
            request_id=request_id,
            processing_time=processing_time,
            confidence=response.confidence
        )
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        logger.error(
            "query_failed",
            request_id=request_id,
            error=str(e),
            processing_time=processing_time
        )
        
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="internal_error",
                message="Erro interno do servidor",
                request_id=request_id
            ).dict()
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de saúde do serviço"""
    
    try:
        # Verificar status do banco vetorial
        db_status = "healthy"
        try:
            collection_count = rag_system.vectorstore._collection.count()
            if collection_count == 0:
                db_status = "empty"
        except Exception:
            db_status = "unhealthy"
        
        # Verificar status do modelo
        model_status = "healthy"
        try:
            # Teste simples do modelo
            test_embedding = rag_system.embeddings.embed_query("test")
            if not test_embedding:
                model_status = "unhealthy"
        except Exception:
            model_status = "unhealthy"
        
        return HealthResponse(
            status="healthy" if db_status == "healthy" and model_status == "healthy" else "degraded",
            version="1.0.0",
            database_status=db_status,
            model_status=model_status
        )
        
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="service_unavailable",
                message="Serviço temporariamente indisponível"
            ).dict()
        )

@router.get("/stats")
async def get_stats(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Estatísticas do sistema"""
    
    user_id = await verify_token(credentials.credentials)
    
    try:
        stats = {
            "documents_count": rag_system.vectorstore._collection.count(),
            "embeddings_model": rag_system.embeddings.model_name,
            "llm_model": getattr(rag_system.llm, "model_name", "unknown"),
            "uptime": time.time() - app.start_time
        }
        
        return stats
        
    except Exception as e:
        logger.error("stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Erro ao obter estatísticas")
```

---

## 3. Validação e tratamento de erros

### 3.1. Middleware de tratamento de erros

```python
# api/middleware.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import structlog
import time
import uuid

logger = structlog.get_logger()

@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Middleware para tratamento global de erros"""
    
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        processing_time = time.time() - start_time
        
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            processing_time=processing_time
        )
        
        return response
        
    except HTTPException as e:
        processing_time = time.time() - start_time
        
        logger.warning(
            "http_exception",
            request_id=request_id,
            status_code=e.status_code,
            detail=e.detail,
            processing_time=processing_time
        )
        
        return JSONResponse(
            status_code=e.status_code,
            content={
                "error": "http_error",
                "message": e.detail,
                "request_id": request_id,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        logger.error(
            "unexpected_error",
            request_id=request_id,
            error=str(e),
            processing_time=processing_time
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "Erro interno do servidor",
                "request_id": request_id,
                "timestamp": time.time()
            }
        )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler para erros de validação"""
    
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.warning(
        "validation_error",
        request_id=request_id,
        errors=exc.errors()
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Dados de entrada inválidos",
            "details": exc.errors(),
            "request_id": request_id,
            "timestamp": time.time()
        }
    )
```

### 3.2. Validadores customizados

```python
# api/validators.py
from fastapi import HTTPException
import re
from typing import List

class QueryValidator:
    """Validador para consultas RAG"""
    
    BLOCKED_PATTERNS = [
        r'\b(hack|crack|bypass|exploit)\b',
        r'\b(password|senha|token|api_key)\b',
        r'<script.*?>.*?</script>',
        r'(delete|drop|truncate)\s+table',
    ]
    
    TOXIC_KEYWORDS = [
        'violência', 'ódio', 'discriminação',
        'illegal', 'drogas', 'armas'
    ]
    
    @classmethod
    def validate_query_safety(cls, query: str) -> List[str]:
        """Valida se a consulta é segura"""
        
        violations = []
        query_lower = query.lower()
        
        # Verificar padrões bloqueados
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                violations.append(f"Padrão bloqueado detectado")
        
        # Verificar palavras tóxicas
        for keyword in cls.TOXIC_KEYWORDS:
            if keyword in query_lower:
                violations.append(f"Conteúdo inadequado detectado")
        
        return violations
    
    @classmethod
    def validate_query_length(cls, query: str) -> None:
        """Valida o tamanho da consulta"""
        
        if len(query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Consulta muito curta (mínimo 3 caracteres)"
            )
        
        if len(query) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Consulta muito longa (máximo 1000 caracteres)"
            )
    
    @classmethod
    def validate_query(cls, query: str) -> List[str]:
        """Validação completa da consulta"""
        
        cls.validate_query_length(query)
        violations = cls.validate_query_safety(query)
        
        if violations:
            raise HTTPException(
                status_code=400,
                detail=f"Consulta rejeitada: {'; '.join(violations)}"
            )
        
        return violations
```

---

## 4. Autenticação e segurança

### 4.1. Sistema de autenticação JWT

```python
# api/auth.py
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import os

# Configurações
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthManager:
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

async def verify_token(token: str) -> str:
    """Verifica e decodifica token JWT"""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise credentials_exception
            
        return user_id
        
    except JWTError:
        raise credentials_exception

# Endpoint de login
@router.post("/auth/login")
async def login(username: str, password: str):
    """Endpoint de autenticação"""
    
    # Aqui você faria a verificação no banco de dados
    # Para este exemplo, usamos credenciais fixas
    if username == "admin" and password == "password":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = AuthManager.create_access_token(
            data={"sub": username}, 
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciais inválidas"
    )
```

### 4.2. Rate limiting

```python
# api/rate_limit.py
from fastapi import HTTPException, Request
from collections import defaultdict, deque
import time
from typing import Dict, Deque

class RateLimiter:
    """Rate limiter simples baseado em memória"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, Deque[float]] = defaultdict(deque)
    
    def is_allowed(self, client_id: str) -> bool:
        """Verifica se o cliente pode fazer uma requisição"""
        
        now = time.time()
        window_start = now - self.window_seconds
        
        # Limpar requisições antigas
        client_requests = self.requests[client_id]
        while client_requests and client_requests[0] < window_start:
            client_requests.popleft()
        
        # Verificar limite
        if len(client_requests) >= self.max_requests:
            return False
        
        # Adicionar requisição atual
        client_requests.append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """Retorna número de requisições restantes"""
        
        now = time.time()
        window_start = now - self.window_seconds
        
        client_requests = self.requests[client_id]
        valid_requests = sum(1 for req_time in client_requests if req_time >= window_start)
        
        return max(0, self.max_requests - valid_requests)

# Instância global
rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)

async def check_rate_limit(request: Request):
    """Middleware de rate limiting"""
    
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": "Muitas requisições. Tente novamente em 1 hora.",
                "remaining": rate_limiter.get_remaining(client_ip)
            }
        )
```

---

## 5. Containerização com Docker

### 5.1. Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Definir working directory
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
COPY requirements_production.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements_production.txt

# Copiar código da aplicação
COPY . .

# Criar usuário não-root
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicialização
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.2. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - LOG_LEVEL=INFO
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./db:/app/db
    depends_on:
      - redis
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rag-api
    restart: unless-stopped

volumes:
  redis_data:
```

### 5.3. Configuração do Nginx

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream rag_api {
        server rag-api:8000;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl;
        server_name your-domain.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000";
        
        location / {
            # Rate limiting
            limit_req zone=api burst=20 nodelay;
            
            # Proxy para a API
            proxy_pass http://rag_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        location /health {
            proxy_pass http://rag_api/health;
            access_log off;
        }
    }
}
```

---

## 6. Configuração e variáveis de ambiente

### 6.1. Configuração centralizada

```python
# config/settings.py
from pydantic import BaseSettings, Field
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Configurações da aplicação"""
    
    # API
    app_name: str = Field(default="RAG API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database
    chroma_db_path: str = Field(default="./db", env="CHROMA_DB_PATH")
    chroma_collection_name: str = Field(default="documents", env="CHROMA_COLLECTION_NAME")
    
    # LLM
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL")
    embeddings_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDINGS_MODEL")
    
    # Security
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=30, env="JWT_EXPIRE_MINUTES")
    
    # CORS
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000"], 
        env="ALLOWED_ORIGINS"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    
    # Redis (para cache e sessões)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Instância global
settings = Settings()
```

### 6.2. Arquivo de environment

```bash
# .env.example
# Copie para .env e preencha os valores

# API Configuration
APP_NAME="RAG API"
APP_VERSION="1.0.0"
DEBUG=false

# Database
CHROMA_DB_PATH="./db"
CHROMA_COLLECTION_NAME="documents"

# LLM Configuration
OPENAI_API_KEY="sk-your-openai-api-key"
LLM_MODEL="gpt-3.5-turbo"
EMBEDDINGS_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Security
JWT_SECRET_KEY="your-very-secret-jwt-key-change-in-production"
JWT_ALGORITHM="HS256"
JWT_EXPIRE_MINUTES=30

# CORS
ALLOWED_ORIGINS='["http://localhost:3000","https://yourdomain.com"]'

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Logging
LOG_LEVEL="INFO"
LOG_FORMAT="json"

# Monitoring
SENTRY_DSN=""
ENABLE_METRICS=true

# Redis
REDIS_URL="redis://localhost:6379"
```

---

## 7. Integração com aplicações externas

### 7.1. Cliente Python

```python
# clients/python_client.py
import requests
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]
    confidence: float
    processing_time: float
    request_id: str

class RAGClient:
    """Cliente Python para a API RAG"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}'
            })
    
    def authenticate(self, username: str, password: str) -> str:
        """Autentica e retorna token"""
        
        response = self.session.post(
            f"{self.base_url}/auth/login",
            data={"username": username, "password": password}
        )
        response.raise_for_status()
        
        token_data = response.json()
        token = token_data["access_token"]
        
        # Atualizar headers com token
        self.session.headers.update({
            'Authorization': f'Bearer {token}'
        })
        
        return token
    
    def query(
        self, 
        query: str, 
        max_results: int = 5,
        threshold: float = 0.7,
        use_guardrails: bool = True,
        conversation_id: Optional[str] = None
    ) -> RAGResponse:
        """Realiza consulta RAG"""
        
        payload = {
            "query": query,
            "max_results": max_results,
            "threshold": threshold,
            "use_guardrails": use_guardrails
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        response = self.session.post(
            f"{self.base_url}/query",
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        
        return RAGResponse(
            answer=data["answer"],
            sources=data["sources"],
            confidence=data["confidence"],
            processing_time=data["processing_time"],
            request_id=data["request_id"]
        )
    
    def health_check(self) -> Dict:
        """Verifica saúde do serviço"""
        
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        
        return response.json()

# Exemplo de uso
if __name__ == "__main__":
    client = RAGClient("http://localhost:8000")
    
    # Autenticar
    client.authenticate("admin", "password")
    
    # Fazer consulta
    result = client.query("O que é machine learning?")
    
    print(f"Resposta: {result.answer}")
    print(f"Confiança: {result.confidence}")
    print(f"Fontes: {len(result.sources)}")
```

### 7.2. Cliente JavaScript/TypeScript

```typescript
// clients/typescript_client.ts
interface RAGQueryRequest {
    query: string;
    max_results?: number;
    threshold?: number;
    use_guardrails?: boolean;
    conversation_id?: string;
}

interface RAGSource {
    content: string;
    similarity: number;
    metadata: Record<string, any>;
}

interface RAGResponse {
    request_id: string;
    query: string;
    answer: string;
    sources: RAGSource[];
    confidence: number;
    processing_time: number;
    timestamp: string;
    guardrails_triggered: string[];
}

class RAGClient {
    private baseUrl: string;
    private token?: string;

    constructor(baseUrl: string, token?: string) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.token = token;
    }

    private async request<T>(
        endpoint: string, 
        options: RequestInit = {}
    ): Promise<T> {
        const url = `${this.baseUrl}${endpoint}`;
        
        const headers: HeadersInit = {
            'Content-Type': 'application/json',
            ...options.headers,
        };

        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }

        const response = await fetch(url, {
            ...options,
            headers,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ 
                message: response.statusText 
            }));
            throw new Error(error.message || `HTTP ${response.status}`);
        }

        return response.json();
    }

    async authenticate(username: string, password: string): Promise<string> {
        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);

        const response = await this.request<{
            access_token: string;
            token_type: string;
            expires_in: number;
        }>('/auth/login', {
            method: 'POST',
            body: formData,
            headers: {}, // Let browser set Content-Type for FormData
        });

        this.token = response.access_token;
        return this.token;
    }

    async query(request: RAGQueryRequest): Promise<RAGResponse> {
        return this.request<RAGResponse>('/query', {
            method: 'POST',
            body: JSON.stringify(request),
        });
    }

    async healthCheck(): Promise<any> {
        return this.request<any>('/health');
    }

    async getStats(): Promise<any> {
        return this.request<any>('/stats');
    }
}

// Exemplo de uso em React
export const useRAG = () => {
    const [client] = useState(() => new RAGClient('http://localhost:8000'));
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    const authenticate = async (username: string, password: string) => {
        try {
            await client.authenticate(username, password);
            setIsAuthenticated(true);
            return true;
        } catch (error) {
            console.error('Authentication failed:', error);
            return false;
        }
    };

    const query = async (queryText: string): Promise<RAGResponse | null> => {
        try {
            return await client.query({ query: queryText });
        } catch (error) {
            console.error('Query failed:', error);
            return null;
        }
    };

    return { authenticate, query, isAuthenticated };
};
```

### 7.3. Integração via webhook

```python
# api/webhooks.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl
import httpx
import structlog

logger = structlog.get_logger()

router = APIRouter()

class WebhookConfig(BaseModel):
    url: HttpUrl
    events: List[str]  # ['query_completed', 'query_failed']
    secret: Optional[str] = None

class WebhookEvent(BaseModel):
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    request_id: str

# Storage simples (em produção, usar banco de dados)
webhooks: Dict[str, WebhookConfig] = {}

@router.post("/webhooks/register")
async def register_webhook(
    webhook: WebhookConfig,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Registra webhook para eventos"""
    
    user_id = await verify_token(credentials.credentials)
    webhook_id = f"{user_id}_{uuid.uuid4()}"
    
    webhooks[webhook_id] = webhook
    
    logger.info(
        "webhook_registered",
        webhook_id=webhook_id,
        user_id=user_id,
        url=str(webhook.url),
        events=webhook.events
    )
    
    return {"webhook_id": webhook_id, "status": "registered"}

async def send_webhook(
    webhook_id: str,
    event: WebhookEvent
):
    """Envia evento via webhook"""
    
    if webhook_id not in webhooks:
        return
    
    webhook = webhooks[webhook_id]
    
    if event.event_type not in webhook.events:
        return
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            
            if webhook.secret:
                import hmac
                import hashlib
                
                payload = event.json()
                signature = hmac.new(
                    webhook.secret.encode(),
                    payload.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                headers["X-Webhook-Signature"] = f"sha256={signature}"
            
            response = await client.post(
                str(webhook.url),
                json=event.dict(),
                headers=headers,
                timeout=10.0
            )
            
            logger.info(
                "webhook_sent",
                webhook_id=webhook_id,
                event_type=event.event_type,
                status_code=response.status_code
            )
            
    except Exception as e:
        logger.error(
            "webhook_failed",
            webhook_id=webhook_id,
            event_type=event.event_type,
            error=str(e)
        )

# Usar em endpoints existentes
async def notify_webhooks(event_type: str, data: dict, request_id: str):
    """Notifica todos os webhooks relevantes"""
    
    event = WebhookEvent(
        event_type=event_type,
        timestamp=datetime.now(),
        data=data,
        request_id=request_id
    )
    
    for webhook_id in webhooks:
        await send_webhook(webhook_id, event)
```

---

## 8. Monitoramento e logs

### 8.1. Logging estruturado

```python
# api/logging.py
import structlog
import logging
from datetime import datetime
import json
import sys

def configure_logging(log_level: str = "INFO", log_format: str = "json"):
    """Configura logging estruturado"""
    
    # Configurar nivel
    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)
    
    # Processadores do structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Middleware de logging
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware para logging de requisições"""
    
    start_time = time.time()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "request_started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get("user-agent"),
        client_ip=request.client.host
    )
    
    response = await call_next(request)
    
    processing_time = time.time() - start_time
    
    logger.info(
        "request_completed",
        request_id=request_id,
        status_code=response.status_code,
        processing_time=processing_time
    )
    
    return response
```

### 8.2. Métricas com Prometheus

```python
# api/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time

# Métricas
request_count = Counter(
    'rag_requests_total',
    'Total requests to RAG API',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'rag_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

query_confidence = Histogram(
    'rag_query_confidence',
    'Confidence score of RAG responses',
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
)

active_requests = Gauge(
    'rag_active_requests',
    'Number of active requests'
)

documents_count = Gauge(
    'rag_documents_total',
    'Total number of documents in database'
)

@router.get("/metrics")
async def metrics():
    """Endpoint de métricas Prometheus"""
    
    # Atualizar métricas dinâmicas
    try:
        doc_count = rag_system.vectorstore._collection.count()
        documents_count.set(doc_count)
    except Exception:
        pass
    
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

async def record_query_metrics(
    request_id: str,
    user_id: str,
    processing_time: float,
    confidence: float,
    num_sources: int
):
    """Registra métricas de consulta"""
    
    query_confidence.observe(confidence)
    
    # Log para análise posterior
    logger.info(
        "query_metrics",
        request_id=request_id,
        user_id=user_id,
        processing_time=processing_time,
        confidence=confidence,
        num_sources=num_sources
    )

# Middleware de métricas
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware para coleta de métricas"""
    
    active_requests.inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Registrar métricas
        processing_time = time.time() - start_time
        
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(processing_time)
        
        return response
        
    finally:
        active_requests.dec()
```

---

## 9. Estratégias de deployment

### 9.1. Script de deploy

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

echo "🚀 Iniciando deploy da RAG API..."

# Verificar dependências
command -v docker >/dev/null 2>&1 || { echo "Docker não encontrado" >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose não encontrado" >&2; exit 1; }

# Verificar variáveis de ambiente
if [ ! -f .env ]; then
    echo "❌ Arquivo .env não encontrado"
    echo "Copie .env.example para .env e configure as variáveis"
    exit 1
fi

# Verificar se OPENAI_API_KEY está definida
source .env
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY não definida no .env"
    exit 1
fi

# Build da imagem
echo "🔨 Construindo imagem Docker..."
docker-compose build

# Testar configuração
echo "🧪 Testando configuração..."
docker-compose config

# Deploy
echo "🚀 Fazendo deploy..."
docker-compose up -d

# Aguardar serviço ficar disponível
echo "⏳ Aguardando serviço ficar disponível..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo "✅ Serviço disponível!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Timeout aguardando serviço"
        docker-compose logs
        exit 1
    fi
    sleep 2
done

# Verificar logs
echo "📋 Últimos logs:"
docker-compose logs --tail=10

echo "✅ Deploy concluído!"
echo "🌐 API disponível em: http://localhost:8000"
echo "📚 Documentação em: http://localhost:8000/docs"
```

### 9.2. Deploy para Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  labels:
    app: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: jwt-secret-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: db-volume
          mountPath: /app/db
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: rag-data-pvc
      - name: db-volume
        persistentVolumeClaim:
          claimName: rag-db-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  jwt-secret-key: <base64-encoded-secret>
```

### 9.3. CI/CD com GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy RAG API

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest tests/ -v --cov=rag_demo
    
    - name: Lint code
      run: |
        pip install flake8 black
        flake8 rag_demo/ api/
        black --check rag_demo/ api/

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production..."
        # Aqui você colocaria os comandos específicos do seu ambiente
        # Por exemplo, kubectl apply ou chamadas para serviços de cloud
```

---

## 10. Conclusão

Neste capítulo, transformamos nosso sistema RAG em uma aplicação de produção completa:

### 10.1. O que conquistamos

✅ **API REST robusta** com FastAPI  
✅ **Validação e tratamento de erros** abrangente  
✅ **Autenticação JWT** e controle de acesso  
✅ **Containerização** com Docker  
✅ **Monitoramento** com logs e métricas  
✅ **Integração** com aplicações externas  
✅ **Deploy automatizado** e CI/CD  

### 10.2. Próximos passos

Para evoluir ainda mais o sistema:

1. **Escalabilidade horizontal** com load balancers
2. **Cache distribuído** com Redis/Memcached
3. **Banco de dados relacional** para metadados
4. **Observabilidade** com tracing distribuído
5. **Testes automatizados** end-to-end
6. **Backup e recovery** dos dados
7. **Multi-tenancy** para múltiplos clientes

### 10.3. Arquitetura final

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Clients   │    │    CDN/     │    │    Load     │
│ (Web/Mobile)│◄──►│   Proxy     │◄──►│  Balancer   │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         ▼                         │
                    │         ┌─────────────────────────────────┐       │
                    │         │         RAG API Instances       │       │
                    │         │    (FastAPI + Gunicorn)        │       │
                    │         └─────────────────────────────────┘       │
                    │                         │                         │
                    │         ┌───────────────┼───────────────┐         │
                    │         ▼               ▼               ▼         │
                    │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
                    │ │  Vector DB  │ │    Redis    │ │   Monitoring │   │
                    │ │ (ChromaDB)  │ │   (Cache)   │ │(Prometheus) │   │
                    │ └─────────────┘ └─────────────┘ └─────────────┘   │
                    └─────────────────────────────────────────────────────┘
```

O sistema agora está pronto para **produção**, **escalável** e **mantível**. Você tem uma base sólida para construir aplicações RAG robustas que podem atender usuários reais com confiabilidade e performance.

No próximo capítulo, exploraremos **casos de uso avançados** e **especializações** do RAG para diferentes domínios e aplicações específicas.

---

**Recursos adicionais:**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [12-Factor App Methodology](https://12factor.net/)
