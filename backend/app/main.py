from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

from app import models, schemas, crud, auth, ai_resume, ai_video, database

app = FastAPI()

# CORS чтобы фронтенд мог обращаться
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # в реальном проекте укажите фронтенд домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(bind=database.engine)

@app.on_event("startup")
def startup_event():
    db = next(database.get_db())
    crud.create_admin_users(db)

@app.post("/register", response_model=schemas.Msg)
def register(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    if crud.get_user_by_username(db, user.username):
        raise HTTPException(status_code=400, detail="Username taken")
    crud.create_user(db, user)
    return {"msg": "User registered"}

@app.post("/token", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = crud.get_user_by_username(db, form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = auth.create_access_token({"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    content = await file.read()
    result = ai_resume.analyze_resume(content)
    return result

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    content = await file.read()
    result = ai_video.analyze_video(content, file.filename)
    return result
