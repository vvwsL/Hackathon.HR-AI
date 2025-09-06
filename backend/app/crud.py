from sqlalchemy.orm import Session
from . import models, schemas, auth

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_admin_users(db: Session):
    # Create employee admin
    emp_username = 'admin_employee'
    if not get_user_by_username(db, emp_username):
        create_user(db, schemas.UserCreate(
            username=emp_username,
            email='employee@company.com',
            password='admin',
            role='employee'
        ))
    # Create HR admin
    hr_username = 'admin_hr'
    if not get_user_by_username(db, hr_username):
        create_user(db, schemas.UserCreate(
            username=hr_username,
            email='hr@company.com',
            password='admin',
            role='hr'
        ))
