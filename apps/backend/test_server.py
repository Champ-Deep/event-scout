from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple test endpoints
@app.post("/register/")
async def register_user(user: dict):
    return {
        "status": "success",
        "message": "User registered successfully",
        "user_id": "test_user_123",
        "name": user.get("name", "Test User"),
        "email": user.get("email", "test@example.com"),
        "is_admin": False
    }

@app.post("/login/")
async def login_user(credentials: dict):
    return {
        "status": "success",
        "message": "Login successful",
        "user_id": "test_user_123",
        "name": "Test User",
        "email": credentials.get("email", "test@example.com"),
        "is_admin": False
    }

@app.get("/auth/linkedin/url")
async def get_linkedin_auth_url():
    return {
        "url": "https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id=test&redirect_uri=http://localhost:8081/callback&state=test"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)