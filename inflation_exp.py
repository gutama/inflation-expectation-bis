import os
import json
import random
import re
import argparse
import traceback
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from litellm import completion

# Load environment variables
load_dotenv()

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Base configuration
DEFAULT_MODEL = "gpt-4.1-mini"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# SQLAlchemy setup
Base = declarative_base()

#######################
# DATABASE MODELS
#######################

class Report(Base):
    """Model for Bank Indonesia reports"""
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    publication_date = Column(DateTime, nullable=False)
    report_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    
    def __repr__(self):
        return f"<Report(title='{self.title}', date='{self.publication_date}')>"


class SurveyData(Base):
    """Model for actual survey data"""
    __tablename__ = "survey_data"
    
    id = Column(Integer, primary_key=True)
    respondent_id = Column(String(50), nullable=False)
    survey_date = Column(DateTime, nullable=False)
    age = Column(Integer)
    gender = Column(String(20))
    education = Column(String(50))
    expenditure_bracket = Column(String(50))
    expenditure = Column(String(50))
    region = Column(String(100))
    province = Column(String(100))
    urban_rural = Column(String(20))
    inflation_expectation = Column(Float)
    confidence_level = Column(Integer)
    
    def __repr__(self):
        return f"<SurveyData(id='{self.id}', respondent_id='{self.respondent_id}')>"


class ExperimentResult(Base):
    """Model for storing experiment results"""
    __tablename__ = "experiment_results"
    
    id = Column(Integer, primary_key=True)
    persona_id = Column(String(50), nullable=False)
    treatment_group = Column(String(50), nullable=False)
    pre_treatment_expectation = Column(Float)
    post_treatment_expectation = Column(Float)
    expectation_change = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<ExperimentResult(persona_id='{self.persona_id}', treatment='{self.treatment_group}')>"


#######################
# DATA LOADING
#######################

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, db_type="sqlite", db_path=None, db_config=None):
        """
        Initialize database connection
        
        Args:
            db_type: Type of database ('sqlite' or 'postgresql')
            db_path: Path to SQLite database file
            db_config: Configuration for PostgreSQL connection
        """
        self.db_type = db_type
        
        try:
            if db_type == "sqlite":
                db_path = db_path or os.path.join(DATA_DIR, "inflation_study.db")
                self.engine = create_engine(f"sqlite:///{db_path}")
            elif db_type == "postgresql":
                if not db_config:
                    raise ValueError("PostgreSQL configuration required")
                pg_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
                self.engine = create_engine(pg_url)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
                
            # Create tables
            Base.metadata.create_all(self.engine)
            
            # Create session maker
            self.Session = sessionmaker(bind=self.engine)
            
        except Exception as e:
            print(f"Database connection error: {e}")
            raise
    
    def load_reports(self, reports_dir: str = None):
        """
        Load Bank Indonesia reports into database
        
        Args:
            reports_dir: Directory containing report files
        """
        reports_dir = reports_dir or os.path.join(DATA_DIR, "reports")
        
        if not os.path.exists(reports_dir):
            print(f"Reports directory not found: {reports_dir}")
            return
            
        session = self.Session()
        try:
            # Simple implementation - in practice would need parsers for different formats
            for filename in os.listdir(reports_dir):
                if filename.endswith(".txt"):
                    filepath = os.path.join(reports_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Parse filename for metadata (example format: REPORT_TYPE_YYYY_MM_DD_TITLE.txt)
                    parts = filename.replace('.txt', '').split('_')
                    report_type = parts[0]
                    year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
                    title = ' '.join(parts[4:])
                    
                    report = Report(
                        title=title,
                        publication_date=datetime(year, month, day),
                        report_type=report_type,
                        content=content
                    )
                    session.add(report)
            
            session.commit()
            print(f"Loaded reports into database")
        except Exception as e:
            session.rollback()
            print(f"Error loading reports: {e}")
        finally:
            session.close()
    
    def load_survey_data(self, survey_file: str = None):
        """
        Load survey data into database
        
        Args:
            survey_file: CSV file containing survey data
        """
        survey_file = survey_file or os.path.join(DATA_DIR, "survey_data.csv")
        
        if not os.path.exists(survey_file):
            print(f"Survey file not found: {survey_file}")
            return
            
        try:
            df = pd.read_csv(survey_file)
            session = self.Session()
            
            for _, row in df.iterrows():
                survey_data = SurveyData(
                    respondent_id=row.get('respondent_id'),
                    survey_date=datetime.strptime(row.get('survey_date'), '%Y-%m-%d'),
                    age=row.get('age'),
                    gender=row.get('gender'),
                    education=row.get('education'),
                    expenditure_bracket=row.get('expenditure_bracket'),
                    expenditure=row.get('expenditure'),
                    region=row.get('region'),
                    province=row.get('province'),
                    urban_rural=row.get('urban_rural'),
                    inflation_expectation=row.get('inflation_expectation'),
                    confidence_level=row.get('confidence_level')
                )
                session.add(survey_data)
            
            session.commit()
            print(f"Loaded survey data into database")
        except Exception as e:
            if 'session' in locals():
                session.rollback()
            print(f"Error loading survey data: {e}")
        finally:
            if 'session' in locals():
                session.close()
    
    def get_recent_reports(self, report_type: str = None, limit: int = 5) -> List[Dict]:
        """
        Get recent reports from database
        
        Args:
            report_type: Type of report to filter by
            limit: Maximum number of reports to return
            
        Returns:
            List of report dictionaries
        """
        session = self.Session()
        try:
            query = session.query(Report).order_by(Report.publication_date.desc())
            
            if report_type:
                query = query.filter(Report.report_type == report_type)
                
            reports = query.limit(limit).all()
            
            return [
                {
                    'id': report.id,
                    'title': report.title,
                    'publication_date': report.publication_date.isoformat(),
                    'report_type': report.report_type,
                    'content': report.content[:500] + '...' if len(report.content) > 500 else report.content
                }
                for report in reports
            ]
        except Exception as e:
            print(f"Error retrieving reports: {e}")
            return []
        finally:
            session.close()
    
    def get_survey_statistics(self) -> Dict:
        """
        Get statistics from survey data
        
        Returns:
            Dictionary of survey statistics
        """
        session = self.Session()
        try:
            # Get demographic breakdowns
            result = {
                'count': session.query(SurveyData).count(),
                'avg_inflation_expectation': session.query(
                    func.avg(SurveyData.inflation_expectation)
                ).scalar(),
                'gender_distribution': {},
                'age_distribution': {},
                'education_distribution': {},
                'region_distribution': {}
            }
            
            # This is simplified - in a real implementation we would use SQLAlchemy's
            # group_by and aggregate functions to get these distributions
            
            return result
        except Exception as e:
            print(f"Error retrieving survey statistics: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    def save_experiment_result(self, result: Dict):
        """
        Save experiment result to database
        
        Args:
            result: Dictionary of experiment result data
        """
        session = self.Session()
        try:
            experiment_result = ExperimentResult(
                persona_id=result['persona_id'],
                treatment_group=result['treatment_group'],
                pre_treatment_expectation=result['pre_treatment_expectation'],
                post_treatment_expectation=result['post_treatment_expectation'],
                expectation_change=result['post_treatment_expectation'] - result['pre_treatment_expectation']
            )
            session.add(experiment_result)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error saving experiment result: {e}")
        finally:
            session.close()


#######################
# PERSONA GENERATION
#######################

@dataclass
class Persona:
    """Class representing a synthetic Indonesian persona"""
    id: str
    quarter: str
    age: int
    gender: str
    education: str
    income: float
    region: str
    province: str
    urban_rural: str
    financial_literacy: int  # 1-10 scale
    media_exposure: int  # 1-10 scale
    risk_attitude: int  # 1-10 scale (1: risk averse, 10: risk seeking)
    expenditure_bracket: str  # Monthly expenditure bracket
    expenditure: float
    
    def to_dict(self) -> Dict:
        """Convert persona to dictionary"""
        return {
            'id': self.id,
            'quarter': self.quarter,
            'age': self.age,
            'gender': self.gender,
            'education': self.education,
            'income': self.income,
            'region': self.region,
            'province': self.province,
            'urban_rural': self.urban_rural,
            'financial_literacy': self.financial_literacy,
            'media_exposure': self.media_exposure,
            'risk_attitude': self.risk_attitude,
            'expenditure_bracket': self.expenditure_bracket,
            'expenditure': self.expenditure
        }
    
    def to_prompt_description(self) -> str:
        """Convert persona to a description for LLM prompts"""
        income_level = "low" if self.income < 5000000 else "middle" if self.income < 15000000 else "high"
        
        return f"""
        You are a {self.age}-year-old {self.gender} living in a {self.urban_rural} area in {self.province}, {self.region} region of Indonesia. 
        We are currently in {self.quarter}.
        Your highest education level is {self.education} and your monthly income is around Rp {self.income:,.0f}, which is considered {income_level} income.
        Your household expenditure bracket is {self.expenditure_bracket} per month.
        Your financial literacy level is {self.financial_literacy}/10 and your exposure to economic news media is {self.media_exposure}/10.
        You have a {self.risk_attitude}/10 risk attitude (where 1 is very cautious and 10 is very comfortable with risk).
        """


class PersonaGenerator:
    """Generates synthetic Indonesian personas based on demographic data"""
    
    def __init__(self):
        """Initialize the persona generator with Indonesian demographic data"""
        # Province-specific demographic data from survey table
        self.province_demographics = { "Maluku": {
            "region": "Sulampua",
            "expenditure": [
                [
                    25.5,
                    "1-2 Juta"
                ],
                [
                    36.5,
                    "2.1-3 Juta"
                ],
                [
                    28.0,
                    "3.1-4 Juta"
                ],
                [
                    5.5,
                    "4.1-5 Juta"
                ],
                [
                    2.5,
                    "5.1-6 Juta"
                ],
                [
                    0.5,
                    "6.1-7 Juta"
                ],
                [
                    0.5,
                    "7.1-8 Juta"
                ],
                [
                    1.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    32.5,
                    "20-30"
                ],
                [
                    32.5,
                    "31-40"
                ],
                [
                    19.5,
                    "41-50"
                ],
                [
                    7.5,
                    "51-60"
                ],
                [
                    8.0,
                    "> 60"
                ]
            ],
            "education": [
                [
                    66.5,
                    "Senior High School"
                ],
                [
                    7.0,
                    "Diploma"
                ],
                [
                    25.5,
                    "Bachelor's Degree"
                ],
                [
                    1.0,
                    "Master's Degree"
                ]
            ]
        },
        "Lampung": {
            "region": "Sumatra",
            "expenditure": [
                [
                    13.0,
                    "1-2 Juta"
                ],
                [
                    13.0,
                    "2.1-3 Juta"
                ],
                [
                    14.0,
                    "3.1-4 Juta"
                ],
                [
                    20.0,
                    "4.1-5 Juta"
                ],
                [
                    20.0,
                    "5.1-6 Juta"
                ],
                [
                    7.0,
                    "6.1-7 Juta"
                ],
                [
                    7.0,
                    "7.1-8 Juta"
                ],
                [
                    6.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    19.5,
                    "20-30"
                ],
                [
                    43.5,
                    "31-40"
                ],
                [
                    24.5,
                    "41-50"
                ],
                [
                    10.5,
                    "51-60"
                ],
                [
                    2.0,
                    "> 60"
                ]
            ],
            "education": [
                [
                    29.0,
                    "Senior High School"
                ],
                [
                    10.5,
                    "Diploma"
                ],
                [
                    52.5,
                    "Bachelor's Degree"
                ],
                [
                    8.0,
                    "Master's Degree"
                ]
            ]
        },
        "West Java": {
            "region": "Java",
            "expenditure": [
                [
                    15.11,
                    "1-2 Juta"
                ],
                [
                    20.67,
                    "2.1-3 Juta"
                ],
                [
                    19.56,
                    "3.1-4 Juta"
                ],
                [
                    19.56,
                    "4.1-5 Juta"
                ],
                [
                    11.56,
                    "5.1-6 Juta"
                ],
                [
                    6.0,
                    "6.1-7 Juta"
                ],
                [
                    4.44,
                    "7.1-8 Juta"
                ],
                [
                    3.11,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    19.11,
                    "20-30"
                ],
                [
                    24.89,
                    "31-40"
                ],
                [
                    32.0,
                    "41-50"
                ],
                [
                    16.67,
                    "51-60"
                ],
                [
                    7.33,
                    "> 60"
                ]
            ],
            "education": [
                [
                    76.22,
                    "Senior High School"
                ],
                [
                    9.56,
                    "Diploma"
                ],
                [
                    12.89,
                    "Bachelor's Degree"
                ],
                [
                    1.33,
                    "Master's Degree"
                ]
            ]
        },
        "Banten": {
            "region": "Java",
            "expenditure": [
                [
                    13.0,
                    "1-2 Juta"
                ],
                [
                    16.0,
                    "2.1-3 Juta"
                ],
                [
                    19.5,
                    "3.1-4 Juta"
                ],
                [
                    19.0,
                    "4.1-5 Juta"
                ],
                [
                    13.5,
                    "5.1-6 Juta"
                ],
                [
                    5.0,
                    "6.1-7 Juta"
                ],
                [
                    8.0,
                    "7.1-8 Juta"
                ],
                [
                    6.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    30.5,
                    "20-30"
                ],
                [
                    27.0,
                    "31-40"
                ],
                [
                    29.0,
                    "41-50"
                ],
                [
                    11.5,
                    "51-60"
                ],
                [
                    2.0,
                    "> 60"
                ]
            ],
            "education": [
                [
                    52.5,
                    "Senior High School"
                ],
                [
                    13.5,
                    "Diploma"
                ],
                [
                    29.5,
                    "Bachelor's Degree"
                ],
                [
                    4.5,
                    "Master's Degree"
                ]
            ]
        },
        "South Kalimantan": {
            "region": "Kalimantan",
            "expenditure": [
                [
                    37.92,
                    "1-2 Juta"
                ],
                [
                    22.92,
                    "2.1-3 Juta"
                ],
                [
                    13.75,
                    "3.1-4 Juta"
                ],
                [
                    10.0,
                    "4.1-5 Juta"
                ],
                [
                    7.5,
                    "5.1-6 Juta"
                ],
                [
                    2.92,
                    "6.1-7 Juta"
                ],
                [
                    2.5,
                    "7.1-8 Juta"
                ],
                [
                    2.5,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    28.75,
                    "20-30"
                ],
                [
                    25.83,
                    "31-40"
                ],
                [
                    28.33,
                    "41-50"
                ],
                [
                    13.75,
                    "51-60"
                ],
                [
                    3.33,
                    "> 60"
                ]
            ],
            "education": [
                [
                    68.33,
                    "Senior High School"
                ],
                [
                    1.67,
                    "Diploma"
                ],
                [
                    21.25,
                    "Bachelor's Degree"
                ],
                [
                    8.75,
                    "Master's Degree"
                ]
            ]
        },
        "Bali": {
            "region": "Bali-Nusa",
            "expenditure": [
                [
                    8.0,
                    "1-2 Juta"
                ],
                [
                    23.0,
                    "2.1-3 Juta"
                ],
                [
                    20.5,
                    "3.1-4 Juta"
                ],
                [
                    11.0,
                    "4.1-5 Juta"
                ],
                [
                    14.0,
                    "5.1-6 Juta"
                ],
                [
                    10.0,
                    "6.1-7 Juta"
                ],
                [
                    6.5,
                    "7.1-8 Juta"
                ],
                [
                    7.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    23.0,
                    "20-30"
                ],
                [
                    25.5,
                    "31-40"
                ],
                [
                    29.5,
                    "41-50"
                ],
                [
                    17.5,
                    "51-60"
                ],
                [
                    4.5,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.5,
                    "Senior High School"
                ],
                [
                    8.0,
                    "Diploma"
                ],
                [
                    24.5,
                    "Bachelor's Degree"
                ],
                [
                    3.0,
                    "Master's Degree"
                ]
            ]
        },
        "Jakarta": {
            "region": "Java",
            "expenditure": [
                [
                    6.29,
                    "1-2 Juta"
                ],
                [
                    8.57,
                    "2.1-3 Juta"
                ],
                [
                    10.57,
                    "3.1-4 Juta"
                ],
                [
                    10.29,
                    "4.1-5 Juta"
                ],
                [
                    6.86,
                    "5.1-6 Juta"
                ],
                [
                    19.43,
                    "6.1-7 Juta"
                ],
                [
                    19.71,
                    "7.1-8 Juta"
                ],
                [
                    18.29,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    23.14,
                    "20-30"
                ],
                [
                    31.71,
                    "31-40"
                ],
                [
                    28.0,
                    "41-50"
                ],
                [
                    9.14,
                    "51-60"
                ],
                [
                    8.0,
                    "> 60"
                ]
            ],
            "education": [
                [
                    38.86,
                    "Senior High School"
                ],
                [
                    29.14,
                    "Diploma"
                ],
                [
                    20.29,
                    "Bachelor's Degree"
                ],
                [
                    11.71,
                    "Master's Degree"
                ]
            ]
        },
        "South Sulawesi": {
            "region": "Sulampua",
            "expenditure": [
                [
                    20.0,
                    "1-2 Juta"
                ],
                [
                    20.0,
                    "2.1-3 Juta"
                ],
                [
                    18.0,
                    "3.1-4 Juta"
                ],
                [
                    14.0,
                    "4.1-5 Juta"
                ],
                [
                    8.0,
                    "5.1-6 Juta"
                ],
                [
                    9.0,
                    "6.1-7 Juta"
                ],
                [
                    7.0,
                    "7.1-8 Juta"
                ],
                [
                    4.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    33.5,
                    "20-30"
                ],
                [
                    25.5,
                    "31-40"
                ],
                [
                    16.5,
                    "41-50"
                ],
                [
                    18.5,
                    "51-60"
                ],
                [
                    6.0,
                    "> 60"
                ]
            ],
            "education": [
                [
                    72.0,
                    "Senior High School"
                ],
                [
                    8.0,
                    "Diploma"
                ],
                [
                    18.0,
                    "Bachelor's Degree"
                ],
                [
                    2.0,
                    "Master's Degree"
                ]
            ]
        },
        "East Java": {
            "region": "Java",
            "expenditure": [
                [
                    12.4,
                    "1-2 Juta"
                ],
                [
                    22.6,
                    "2.1-3 Juta"
                ],
                [
                    25.8,
                    "3.1-4 Juta"
                ],
                [
                    19.8,
                    "4.1-5 Juta"
                ],
                [
                    13.4,
                    "5.1-6 Juta"
                ],
                [
                    3.4,
                    "6.1-7 Juta"
                ],
                [
                    1.6,
                    "7.1-8 Juta"
                ],
                [
                    1.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    18.2,
                    "20-30"
                ],
                [
                    33.6,
                    "31-40"
                ],
                [
                    26.4,
                    "41-50"
                ],
                [
                    18.2,
                    "51-60"
                ],
                [
                    3.6,
                    "> 60"
                ]
            ],
            "education": [
                [
                    82.4,
                    "Senior High School"
                ],
                [
                    5.0,
                    "Diploma"
                ],
                [
                    9.6,
                    "Bachelor's Degree"
                ],
                [
                    3.0,
                    "Master's Degree"
                ]
            ]
        },
        "North Sulawesi": {
            "region": "Sulampua",
            "expenditure": [
                [
                    14.0,
                    "1-2 Juta"
                ],
                [
                    27.5,
                    "2.1-3 Juta"
                ],
                [
                    22.5,
                    "3.1-4 Juta"
                ],
                [
                    14.0,
                    "4.1-5 Juta"
                ],
                [
                    3.5,
                    "5.1-6 Juta"
                ],
                [
                    11.0,
                    "6.1-7 Juta"
                ],
                [
                    6.5,
                    "7.1-8 Juta"
                ],
                [
                    1.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    15.5,
                    "20-30"
                ],
                [
                    21.5,
                    "31-40"
                ],
                [
                    29.5,
                    "41-50"
                ],
                [
                    26.0,
                    "51-60"
                ],
                [
                    7.5,
                    "> 60"
                ]
            ],
            "education": [
                [
                    75.0,
                    "Senior High School"
                ],
                [
                    2.0,
                    "Diploma"
                ],
                [
                    21.5,
                    "Bachelor's Degree"
                ],
                [
                    1.5,
                    "Master's Degree"
                ]
            ]
        },
        "West Nusa Tenggara": {
            "region": "Bali-Nusa",
            "expenditure": [
                [
                    26.5,
                    "1-2 Juta"
                ],
                [
                    22.5,
                    "2.1-3 Juta"
                ],
                [
                    22.5,
                    "3.1-4 Juta"
                ],
                [
                    13.0,
                    "4.1-5 Juta"
                ],
                [
                    7.0,
                    "5.1-6 Juta"
                ],
                [
                    4.5,
                    "6.1-7 Juta"
                ],
                [
                    2.0,
                    "7.1-8 Juta"
                ],
                [
                    2.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    27.0,
                    "20-30"
                ],
                [
                    34.0,
                    "31-40"
                ],
                [
                    25.0,
                    "41-50"
                ],
                [
                    11.0,
                    "51-60"
                ],
                [
                    3.0,
                    "> 60"
                ]
            ],
            "education": [
                [
                    68.0,
                    "Senior High School"
                ],
                [
                    6.5,
                    "Diploma"
                ],
                [
                    23.5,
                    "Bachelor's Degree"
                ],
                [
                    2.0,
                    "Master's Degree"
                ]
            ]
        },
        "North Sumatra": {
            "region": "Sumatra",
            "expenditure": [
                [
                    15.87,
                    "1-2 Juta"
                ],
                [
                    27.62,
                    "2.1-3 Juta"
                ],
                [
                    35.56,
                    "3.1-4 Juta"
                ],
                [
                    11.75,
                    "4.1-5 Juta"
                ],
                [
                    4.13,
                    "5.1-6 Juta"
                ],
                [
                    2.22,
                    "6.1-7 Juta"
                ],
                [
                    1.27,
                    "7.1-8 Juta"
                ],
                [
                    1.59,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    11.75,
                    "20-30"
                ],
                [
                    29.21,
                    "31-40"
                ],
                [
                    34.29,
                    "41-50"
                ],
                [
                    19.05,
                    "51-60"
                ],
                [
                    5.71,
                    "> 60"
                ]
            ],
            "education": [
                [
                    75.87,
                    "Senior High School"
                ],
                [
                    6.67,
                    "Diploma"
                ],
                [
                    15.56,
                    "Bachelor's Degree"
                ],
                [
                    1.9,
                    "Master's Degree"
                ]
            ]
        },
        "West Sumatra": {
            "region": "Sumatra",
            "expenditure": [
                [
                    8.0,
                    "1-2 Juta"
                ],
                [
                    14.5,
                    "2.1-3 Juta"
                ],
                [
                    17.0,
                    "3.1-4 Juta"
                ],
                [
                    16.5,
                    "4.1-5 Juta"
                ],
                [
                    14.0,
                    "5.1-6 Juta"
                ],
                [
                    11.0,
                    "6.1-7 Juta"
                ],
                [
                    10.0,
                    "7.1-8 Juta"
                ],
                [
                    9.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    17.0,
                    "20-30"
                ],
                [
                    30.0,
                    "31-40"
                ],
                [
                    26.0,
                    "41-50"
                ],
                [
                    19.0,
                    "51-60"
                ],
                [
                    8.0,
                    "> 60"
                ]
            ],
            "education": [
                [
                    70.5,
                    "Senior High School"
                ],
                [
                    8.0,
                    "Diploma"
                ],
                [
                    18.0,
                    "Bachelor's Degree"
                ],
                [
                    3.5,
                    "Master's Degree"
                ]
            ]
        },
        "South Sumatra": {
            "region": "Sumatra",
            "expenditure": [
                [
                    10.33,
                    "1-2 Juta"
                ],
                [
                    29.67,
                    "2.1-3 Juta"
                ],
                [
                    24.67,
                    "3.1-4 Juta"
                ],
                [
                    15.33,
                    "4.1-5 Juta"
                ],
                [
                    6.67,
                    "5.1-6 Juta"
                ],
                [
                    5.0,
                    "6.1-7 Juta"
                ],
                [
                    4.33,
                    "7.1-8 Juta"
                ],
                [
                    4.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    26.67,
                    "20-30"
                ],
                [
                    27.33,
                    "31-40"
                ],
                [
                    23.33,
                    "41-50"
                ],
                [
                    17.0,
                    "51-60"
                ],
                [
                    5.67,
                    "> 60"
                ]
            ],
            "education": [
                [
                    38.0,
                    "Senior High School"
                ],
                [
                    12.33,
                    "Diploma"
                ],
                [
                    46.33,
                    "Bachelor's Degree"
                ],
                [
                    3.33,
                    "Master's Degree"
                ]
            ]
        },
        "Bangka Belitung": {
            "region": "Sumatra",
            "expenditure": [
                [
                    20.0,
                    "1-2 Juta"
                ],
                [
                    13.0,
                    "2.1-3 Juta"
                ],
                [
                    4.5,
                    "3.1-4 Juta"
                ],
                [
                    4.0,
                    "4.1-5 Juta"
                ],
                [
                    23.5,
                    "5.1-6 Juta"
                ],
                [
                    16.0,
                    "6.1-7 Juta"
                ],
                [
                    11.0,
                    "7.1-8 Juta"
                ],
                [
                    8.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.0,
                    "20-30"
                ],
                [
                    23.0,
                    "31-40"
                ],
                [
                    26.5,
                    "41-50"
                ],
                [
                    17.0,
                    "51-60"
                ],
                [
                    9.5,
                    "> 60"
                ]
            ],
            "education": [
                [
                    84.0,
                    "Senior High School"
                ],
                [
                    4.0,
                    "Diploma"
                ],
                [
                    10.5,
                    "Bachelor's Degree"
                ],
                [
                    1.5,
                    "Master's Degree"
                ]
            ]
        },
        "West Kalimantan": {
            "region": "Kalimantan",
            "expenditure": [
                [
                    45.5,
                    "1-2 Juta"
                ],
                [
                    19.5,
                    "2.1-3 Juta"
                ],
                [
                    10.0,
                    "3.1-4 Juta"
                ],
                [
                    10.0,
                    "4.1-5 Juta"
                ],
                [
                    7.5,
                    "5.1-6 Juta"
                ],
                [
                    3.0,
                    "6.1-7 Juta"
                ],
                [
                    1.5,
                    "7.1-8 Juta"
                ],
                [
                    3.0,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    41.5,
                    "20-30"
                ],
                [
                    23.5,
                    "31-40"
                ],
                [
                    17.5,
                    "41-50"
                ],
                [
                    9.5,
                    "51-60"
                ],
                [
                    8.0,
                    "> 60"
                ]
            ],
            "education": [
                [
                    75.5,
                    "Senior High School"
                ],
                [
                    6.0,
                    "Diploma"
                ],
                [
                    17.5,
                    "Bachelor's Degree"
                ],
                [
                    1.0,
                    "Master's Degree"
                ]
            ]
        },
        "East Kalimantan": {
            "region": "Kalimantan",
            "expenditure": [
                [
                    1.0,
                    "1-2 Juta"
                ],
                [
                    4.0,
                    "2.1-3 Juta"
                ],
                [
                    40.0,
                    "3.1-4 Juta"
                ],
                [
                    6.0,
                    "4.1-5 Juta"
                ],
                [
                    16.0,
                    "5.1-6 Juta"
                ],
                [
                    14.5,
                    "6.1-7 Juta"
                ],
                [
                    15.0,
                    "7.1-8 Juta"
                ],
                [
                    3.5,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    28.0,
                    "20-30"
                ],
                [
                    37.5,
                    "31-40"
                ],
                [
                    21.0,
                    "41-50"
                ],
                [
                    10.5,
                    "51-60"
                ],
                [
                    3.0,
                    "> 60"
                ]
            ],
            "education": [
                [
                    70.5,
                    "Senior High School"
                ],
                [
                    11.0,
                    "Diploma"
                ],
                [
                    16.5,
                    "Bachelor's Degree"
                ],
                [
                    2.0,
                    "Master's Degree"
                ]
            ]
        },
        "Central Java": {
            "region": "Java",
            "expenditure": [
                [
                    22.0,
                    "1-2 Juta"
                ],
                [
                    24.67,
                    "2.1-3 Juta"
                ],
                [
                    25.67,
                    "3.1-4 Juta"
                ],
                [
                    15.0,
                    "4.1-5 Juta"
                ],
                [
                    5.33,
                    "5.1-6 Juta"
                ],
                [
                    2.0,
                    "6.1-7 Juta"
                ],
                [
                    3.0,
                    "7.1-8 Juta"
                ],
                [
                    2.33,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    39.0,
                    "20-30"
                ],
                [
                    31.67,
                    "31-40"
                ],
                [
                    16.33,
                    "41-50"
                ],
                [
                    9.0,
                    "51-60"
                ],
                [
                    4.0,
                    "> 60"
                ]
            ],
            "education": [
                [
                    50.33,
                    "Senior High School"
                ],
                [
                    8.67,
                    "Diploma"
                ],
                [
                    36.0,
                    "Bachelor's Degree"
                ],
                [
                    5.0,
                    "Master's Degree"
                ]
            ]
        },
        "Riau": {
            "region": "Sumatra",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "Yogyakarta": {
            "region": "Java",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "Central Kalimantan": {
            "region": "Kalimantan",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "Gorontalo": {
            "region": "Sulampua",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "Central Sulawesi": {
            "region": "Sulampua",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "Southeast Sulawesi": {
            "region": "Sulampua",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "West Papua": {
            "region": "Sulampua",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "East Nusa Tenggara": {
            "region": "Bali-Nusa",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "Riau Islands": {
            "region": "Sumatra",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "Jambi": {
            "region": "Sumatra",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "Bengkulu": {
            "region": "Sumatra",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "North Maluku": {
            "region": "Sulampua",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "West Sulawesi": {
            "region": "Sulampua",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        },
        "Aceh": {
            "region": "Sumatra",
            "expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]
        }
    }
        
        # Default distributions for provinces not in the detailed list
        self.default_distributions = {"expenditure": [
                [
                    16.73,
                    "1-2 Juta"
                ],
                [
                    20.62,
                    "2.1-3 Juta"
                ],
                [
                    21.12,
                    "3.1-4 Juta"
                ],
                [
                    13.77,
                    "4.1-5 Juta"
                ],
                [
                    10.08,
                    "5.1-6 Juta"
                ],
                [
                    7.09,
                    "6.1-7 Juta"
                ],
                [
                    5.99,
                    "7.1-8 Juta"
                ],
                [
                    4.6,
                    "> 8 Juta"
                ]
            ],
            "age": [
                [
                    24.6,
                    "20-30"
                ],
                [
                    29.41,
                    "31-40"
                ],
                [
                    25.74,
                    "41-50"
                ],
                [
                    14.74,
                    "51-60"
                ],
                [
                    5.52,
                    "> 60"
                ]
            ],
            "education": [
                [
                    64.77,
                    "Senior High School"
                ],
                [
                    9.17,
                    "Diploma"
                ],
                [
                    22.32,
                    "Bachelor's Degree"
                ],
                [
                    3.74,
                    "Master's Degree"
                ]
            ]        }
        
        # From Bank Indonesia April 2025 consumer survey
        self.gender_distribution = { 
            'Male': 0.389,
            'Female': 0.611
        }
        
        # Urban/rural distribution
        self.urban_rural_distribution = {
            'Urban': 0.56,
            'Rural': 0.44
        }

        # Regional data
        self.regions = {
            'Java': ['Jakarta', 'West Java', 'Central Java', 'East Java', 'Yogyakarta', 'Banten'],
            'Sumatra': ['Aceh', 'North Sumatra', 'West Sumatra', 'Riau', 'Jambi', 'South Sumatra', 
                       'Bengkulu', 'Lampung', 'Bangka Belitung', 'Riau Islands'],
            'Kalimantan': ['West Kalimantan', 'Central Kalimantan', 'South Kalimantan', 'East Kalimantan', 
                          'North Kalimantan'],
            'Bali-Nusa': ['Bali', 'West Nusa Tenggara', 'East Nusa Tenggara'],
            'Sulampua': [
                'North Sulawesi', 'Central Sulawesi', 'South Sulawesi', 'Southeast Sulawesi', 'Gorontalo', 'West Sulawesi',
                'Maluku', 'North Maluku', 'Papua', 'West Papua'
            ]
        }

        # Region population ratios (approximate)
        self.region_distribution = {
            'Java': 0.56,
            'Sumatra': 0.22,
            'Kalimantan': 0.06,
            'Bali-Nusa': 0.05,
            'Sulampua': 0.11  # Sulawesi + Maluku-Papua 0.07 + 0.04
        }
        
    def sample_from_distribution(self, distribution: List[Tuple[float, str]]) -> str:
        """Sample from a weighted distribution"""
        weights, values = zip(*distribution)
        return random.choices(values, weights=weights)[0]
    
    def generate_region_and_province(self) -> Tuple[str, str]:
        """Generate region and province"""
        regions = list(self.region_distribution.keys())
        weights = list(self.region_distribution.values())

        region = random.choices(regions, weights)[0]
        
        province = random.choice(self.regions[region])
        
        return province, region
    
    def generate_expenditure_bracket(self, province: str) -> str:
        """Generate expenditure bracket based on province distribution"""
        province_data = self.province_demographics.get(province)
        if province_data:
            return self.sample_from_distribution(province_data['expenditure'])
        else:
            return self.sample_from_distribution(self.default_distributions['expenditure'])
    
    def generate_education_from_province(self, province: str) -> str:
        """Generate education level based on province distribution"""
        province_data = self.province_demographics.get(province)
        if province_data:
            return self.sample_from_distribution(province_data['education'])
        else:
            return self.sample_from_distribution(self.default_distributions['education'])
    
    def generate_age_from_province(self, province: str) -> int:
        """Generate age based on province distribution"""
        province_data = self.province_demographics.get(province)
        if province_data:
            age_bracket = self.sample_from_distribution(province_data['age'])
        else:
            age_bracket = self.sample_from_distribution(self.default_distributions['age'])
        
        # Convert age bracket to actual age
        age_ranges = {
            '20-30': (20, 30),
            '31-40': (31, 40),
            '41-50': (41, 50),
            '51-60': (51, 60),
            '> 60': (61, 80)
        }
        
        min_age, max_age = age_ranges[age_bracket]
        return random.randint(min_age, max_age)
    
    def generate_expenditure(self, expenditure_bracket: str) -> float:
        """Generate expenditure based on expenditure bracket"""
        expenditure_range = {
            '1-2 Juta': (1_000_000, 2_000_000),
            '2.1-3 Juta': (2_000_000, 3_000_000),
            '3.1-4 Juta': (3_100_000, 4_000_000),
            '4.1-5 Juta': (4_100_000, 5_000_000),
            '5.1-6 Juta': (5_100_000, 6_000_000),
            '6.1-7 Juta': (6_100_000, 7_000_000),
            '7.1-8 Juta': (7_100_000, 8_000_000),
            '> 8 Juta': (8_100_000, 15_000_000)
        }

        exp_group = expenditure_range[expenditure_bracket]

        return np.random.uniform(exp_group[0], exp_group[1])
    
    def generate_income_from_expenditure(self, expenditure_bracket: str, education: str, urban_rural: str) -> float:
        """Generate income based on expenditure bracket, education, and location"""
        # Map expenditure brackets to income ranges (in IDR)
        expenditure_to_income = {
            '1-2 Juta': (1_500_000, 3_000_000),
            '2.1-3 Juta': (2_500_000, 4_500_000),
            '3.1-4 Juta': (3_500_000, 6_000_000),
            '4.1-5 Juta': (4_500_000, 7_500_000),
            '5.1-6 Juta': (5_500_000, 9_000_000),
            '6.1-7 Juta': (6_500_000, 10_500_000),
            '7.1-8 Juta': (7_500_000, 12_000_000),
            '> 8 Juta': (8_500_000, 20_000_000)
        }
        
        base_min, base_max = expenditure_to_income[expenditure_bracket]
        
        # Adjust for education (higher education = potentially higher income)
        education_multipliers = {
            'Senior High School': 1.0,
            'Diploma': 1.2,
            "Bachelor's Degree": 1.5,
            "Master's Degree": 2.0
        }
        
        multiplier = education_multipliers.get(education, 1.0)
        
        # Adjust for urban/rural
        location_multiplier = 1.2 if urban_rural == 'Urban' else 0.9
        
        # Generate income within the range
        income_min = base_min * multiplier * location_multiplier
        income_max = base_max * multiplier * location_multiplier
        
        return random.uniform(income_min, income_max)
    
    def generate_gender(self) -> str:
        """Generate gender based on distribution"""
        return random.choices(
            list(self.gender_distribution.keys()),
            weights=list(self.gender_distribution.values())
        )[0]
    
    def generate_urban_rural(self, province: str) -> str:
        """Generate urban/rural status with consideration for province"""
        # Adjust probabilities based on province
        urban_prob = self.urban_rural_distribution['Urban']
        
        # Higher urban probability for Jakarta and major provinces
        if province == 'Jakarta':
            urban_prob = 0.95
        elif province in ['West Java', 'East Java', 'Central Java', 'Bali']:
            urban_prob = 0.65
        # Lower urban probability for certain provinces
        elif province in ['West Papua', 'East Nusa Tenggara', 'Maluku']:
            urban_prob = 0.30
            
        return 'Urban' if random.random() < urban_prob else 'Rural'
    
    def generate_financial_literacy(self, education: str, income: float) -> int:
        """Generate financial literacy score (1-10)"""
        # Base by education level
        edu_scores = {
            'Senior High School': 5,
            'Diploma': 6,
            "Bachelor's Degree": 7,
            "Master's Degree": 8
        }
        
        # Adjust by income (higher income = more financial exposure)
        income_factor = min(3, income / 10_000_000)
        
        base_score = edu_scores.get(education, 5)
        adjusted_score = base_score + (random.random() * 2 - 1) + (income_factor * 0.5)
        
        return max(1, min(10, int(adjusted_score)))
    
    def generate_media_exposure(self, education: str, urban_rural: str) -> int:
        """Generate media exposure score (1-10)"""
        # Base by education
        base_score = {
            'Senior High School': 5,
            'Diploma': 6,
            "Bachelor's Degree": 7,
            "Master's Degree": 8
        }.get(education, 5)
        
        # Urban dwellers have higher media exposure
        location_factor = 1.2 if urban_rural == 'Urban' else 0.8
        
        score = base_score * location_factor + (random.random() * 2 - 1)
        
        return max(1, min(10, int(score)))
    
    def generate_risk_attitude(self, age: int, gender: str) -> int:
        """Generate risk attitude score (1-10)"""
        # Score normally distributed and determined from Sahm (2007)
        score = np.random.normal(5.5, 1.8)
            
        return max(1, min(10, int(score)))
    
    def generate_persona(self, quarter: str) -> Persona:
        """Generate a complete synthetic persona"""
        # Generate ID with timestamp and random string
        persona_id = f"P{int(datetime.now().timestamp())}{random.randint(1000, 9999)}"
        
        # Generate province-based demographics first
        province, region = self.generate_region_and_province()
        
        # Generate other attributes based on province-specific distributions
        age = self.generate_age_from_province(province)
        gender = self.generate_gender()
        education = self.generate_education_from_province(province)
        expenditure_bracket = self.generate_expenditure_bracket(province)
        expenditure = self.generate_expenditure(expenditure_bracket)
        urban_rural = self.generate_urban_rural(province)
        income = self.generate_income_from_expenditure(expenditure_bracket, education, urban_rural)
        financial_literacy = self.generate_financial_literacy(education, income)
        media_exposure = self.generate_media_exposure(education, urban_rural)
        risk_attitude = self.generate_risk_attitude(age, gender)
        
        return Persona(
            id=persona_id,
            quarter=quarter,
            age=age,
            gender=gender,
            education=education,
            income=income,
            region=region,
            province=province,
            urban_rural=urban_rural,
            financial_literacy=financial_literacy,
            media_exposure=media_exposure,
            risk_attitude=risk_attitude,
            expenditure_bracket=expenditure_bracket,
            expenditure=expenditure
        )
    
    def generate_personas(self, count: int, current_quarter: str) -> List[Persona]:
        """Generate multiple personas"""
        return [self.generate_persona(quarter=current_quarter) for _ in range(count)]


#######################
# LLM AGENT MODELING
#######################

class LLMAgent:
    """Agent that simulates Indonesian household inflation expectations using LLMs"""
    
    def __init__(self, persona: Persona, model: str = DEFAULT_MODEL):
        """
        Initialize agent with persona
        
        Args:
            persona: The synthetic persona this agent represents
            model: LLM model to use
        """
        self.persona = persona
        self.model = model
        self.memory = {
            'pre_treatment_expectation': None,
            'post_treatment_expectation': None,
            'treatment_received': None,
            'reasoning': None
        }
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for LLM based on persona"""
        return f"""
        You are simulating the economic thinking and inflation expectations of an Indonesian citizen with the following characteristics:
        
        {self.persona.to_prompt_description()}
        
        Your task is to realistically role-play this person's economic views and expectations, considering their background.
        
        Some important economic context about Indonesia:
        - Indonesia is the largest economy in Southeast Asia
        - Indonesia's central bank is Bank Indonesia (BI)
        - In recent years, inflation has typically been in the 2-5% range
        - The economy is diverse but still reliant on natural resources and agriculture in many regions
        - Bank Indonesia sets interest rate policy to control inflation
        
        Please respond as this person would, considering their education level, financial literacy, and background.
        """
    
    def predict_inflation_expectation(self, message: str) -> Dict:
        """
        Query the LLM to simulate the persona's inflation expectation
        
        Args:
            message: Prompt message to send to the LLM
            
        Returns:
            Dictionary with inflation expectation and confidence
        """
        try:
            system_prompt = self.get_system_prompt()
            
            # Request both a numerical estimate and reasoning
            full_message = f"""
            {message}
            
            Please provide your response in JSON format with the following fields:
            1. inflation_expectation: Your numerical estimate of inflation (as a percentage)
            2. confidence: How confident you are in this estimate (1-10 scale)
            3. reasoning: A brief explanation of your thinking
            
            Example format:
            {{
                "inflation_expectation": 4.5,
                "confidence": 7,
                "reasoning": "I think inflation will be moderate because..."
            }}
            
            Remember to respond as the Indonesian person described, with their level of economic understanding.
            """
            
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_message}
                ],
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    # If JSON parsing fails, estimate values from text
                    print(f"JSON parsing failed, estimating from text: {content}")
                    import re
                    
                    # Look for percentage patterns
                    inflation_matches = re.findall(r'(\d+\.?\d*)%', content)
                    if inflation_matches:
                        inflation = float(inflation_matches[0])
                    else:
                        inflation = 4.0  # Fallback value
                        
                    # Default confidence if not found
                    confidence = 5
                    confidence_matches = re.findall(r'confidence[:\s]+(\d+)', content.lower())
                    if confidence_matches:
                        confidence = int(confidence_matches[0])
                    
                    result = {
                        "inflation_expectation": inflation,
                        "confidence": confidence,
                        "reasoning": content
                    }
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                result = {
                    "inflation_expectation": 4.0,  # Fallback value
                    "confidence": 5,
                    "reasoning": content
                }
            
            return result
            
        except Exception as e:
            print(f"Error in LLM request: {e}")
            # Fallback to reasonable default
            return {
                "inflation_expectation": 4.0,  # Indonesian inflation has historically been around 3-5%
                "confidence": 5,
                "reasoning": "Error in processing response"
            }
    
    def pre_treatment_survey(self) -> Dict:
        """
        Conduct pre-treatment survey to measure baseline inflation expectations
        
        Returns:
            Dictionary with survey results
        """
        message = """
        I'd like to ask you about your expectations for inflation in Indonesia over the next year.
        
        Given your understanding of the economy and recent trends, what percentage do you expect prices 
        to increase by in the next 12 months?
        """
        
        result = self.predict_inflation_expectation(message)
        self.memory['pre_treatment_expectation'] = result
        
        return result
    
    def apply_treatment(self, treatment_type: str, treatment_content: str) -> None:
        """
        Apply information treatment to the agent
        
        Args:
            treatment_type: Type of treatment (e.g., 'control', 'current_inflation', etc.)
            treatment_content: Content of the treatment
        """
        self.memory['treatment_received'] = {
            'type': treatment_type,
            'content': treatment_content
        }
    
    def post_treatment_survey(self) -> Dict:
        """
        Conduct post-treatment survey to measure updated inflation expectations
        
        Returns:
            Dictionary with survey results
        """
        treatment_info = ""
        if self.memory['treatment_received']:
            treatment_info = f"""
            You recently received the following information about the economy:
            
            {self.memory['treatment_received']['content']}
            
            Considering this new information along with your prior knowledge:
            """
        
        message = f"""
        {treatment_info}
        
        What percentage do you now expect prices to increase by in Indonesia over the next 12 months?
        """
        
        result = self.predict_inflation_expectation(message)
        self.memory['post_treatment_expectation'] = result
        
        return result
    
    def get_expectation_change(self) -> Dict:
        """
        Calculate the change in inflation expectations
        
        Returns:
            Dictionary with expectation change data
        """
        pre = self.memory['pre_treatment_expectation']
        post = self.memory['post_treatment_expectation']
        
        if not pre or not post:
            return {'error': 'Missing pre or post treatment data'}
        
        change = post['inflation_expectation'] - pre['inflation_expectation']
        
        return {
            'quarter': self.persona.quarter,
            'persona_id': self.persona.id,
            'treatment_group': self.memory['treatment_received']['type'] if self.memory['treatment_received'] else 'none',
            'pre_treatment_expectation': pre['inflation_expectation'],
            'post_treatment_expectation': post['inflation_expectation'],
            'expectation_change': change,
            'pre_confidence': pre['confidence'],
            'post_confidence': post['confidence'],
            'pre_reasoning': pre['reasoning'],
            'post_reasoning': post['reasoning']
        }

#######################
# EXPERIMENT IMPLEMENTATION
#######################

class ExperimentManager:
    """Manager for experiment execution"""

    # Updated treatment types based on research proposal Table 1
    TREATMENT_TYPES = [
        'control',
        'current_inflation_target',
        'media_narrative',
        'full_policy_context',
        'policy_rate_decision'
    ]

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.persona_generator = PersonaGenerator()
        self.economic_context = None

    def set_economic_context(self, context: Dict):
        """Set economic context from a CSV for current run"""
        self.economic_context = context
    
    def generate_treatments(self) -> Dict[str, str]:
        """
        Generate treatment content for different groups
        
        Returns:
            Dictionary mapping treatment types to content
        """
        context = self.economic_context
        
        treatments = {
            'control': """
                Thank you for participating in our economic survey. 
                Please provide your inflation forecast for the next 12 months.
            """,
            'current_inflation_target': f"""
                According to the latest data from Bank Indonesia, the current inflation rate is {context['current_inflation']}%. 
                The central bank aims to keep annual inflation within its target range of {context['inflation_target']}.
            """,
            'media_narrative': f"""
                According to the latest data from Bank Indonesia, the current inflation rate is {context['current_inflation']}%. 
                The central bank aims to keep annual inflation within its target range of {context['inflation_target']}.
                {context['media_narrative']}
            """,
            'full_policy_context': f"""
                Bank Indonesia has announced its policy stance: 
                - Current inflation: {context['current_inflation']}%
                - Policy interest rate: {context['policy_rate']}%
                - Inflation target: {context['inflation_target']}
                - GDP growth: {context['gdp_growth']}%
                - Economic outlook: {context['economic_outlook']}
                - Additional measures: New price controls on food and energy.
            """,
            'policy_rate_decision': f"""
                Bank Indonesia has changed its interest rate to {context['policy_rate']}% to help control inflation. 
                This decision signals an adjustment of monetary policy.
                An increase in interest rate signals a tightening of the economy, aiming to increase savings and reduce spending. 
                A decrease in interest rate signals easing of the economy, aiming to stimulate economic activity.
            """
        }
        
        return treatments
    
    def assign_treatment_groups(self, count_per_group: int = 30) -> List[str]:
        """
        Create balanced treatment groups with exactly count_per_group personas in each
        
        Args:
            count_per_group: Number of personas per treatment group (default: 30)
            
        Returns:
            List of treatment group assignments
        """
        assignments = []
        for treatment in self.TREATMENT_TYPES:
            assignments.extend([treatment] * count_per_group)
        random.shuffle(assignments)
        print(f"Created treatment assignments: {len(assignments)} total, {count_per_group} per group")
        return assignments
    
    def run_experiment(self, personas_per_group: int = 30, model: str = DEFAULT_MODEL, quarter=None) -> Dict:
        """
        Run the full experiment with balanced treatment groups
        
        Args:
            personas_per_group: Number of personas per treatment group (default: 30)
            model: LLM model to use
            
        Returns:
            Dictionary with experiment results
        """
        total_personas = personas_per_group * len(self.TREATMENT_TYPES)
        print(f"Starting experiment with {personas_per_group} personas per group ({total_personas} total) using {model}...")
        personas = self.persona_generator.generate_personas(total_personas, current_quarter=quarter)
        print(f"Generated {len(personas)} personas")
        treatments = self.generate_treatments()
        treatment_assignments = self.assign_treatment_groups(personas_per_group)
        group_counts = {}
        for t in treatment_assignments:
            group_counts[t] = group_counts.get(t, 0) + 1
        print(f"Treatment group counts: {group_counts}")
        results = []
        for i, persona in enumerate(personas):
            print(f"Running experiment for persona {i+1}/{len(personas)} (ID: {persona.id})...")
            agent = LLMAgent(persona, model=model)
            pre_result = agent.pre_treatment_survey()
            print(f"  Pre-treatment expectation: {pre_result['inflation_expectation']}%")
            treatment_type = treatment_assignments[i]
            agent.apply_treatment(treatment_type, treatments[treatment_type])
            print(f"  Applied treatment: {treatment_type}")
            post_result = agent.post_treatment_survey()
            print(f"  Post-treatment expectation: {post_result['inflation_expectation']}%")
            result_data = agent.get_expectation_change()
            result_data['persona'] = persona.to_dict()
            results.append(result_data)
            self.db_manager.save_experiment_result({
                'persona_id': persona.id,
                'treatment_group': treatment_type,
                'pre_treatment_expectation': pre_result['inflation_expectation'],
                'post_treatment_expectation': post_result['inflation_expectation']
            })
        print(f"Experiment completed with {len(results)} results")
        return {
            'timestamp': datetime.now().isoformat(),
            'persona_count': len(personas),
            'model': model,
            'results': results
        }


#######################
# DATA ANALYSIS
#######################

class DataAnalyzer:
    """Analyzes experimental results"""
    
    def __init__(self, results: List[Dict] = None, results_file: str = None):
        """
        Initialize data analyzer
        
        Args:
            results: List of experiment results
            results_file: Path to JSON file with results
        """
        if results:
            self.results = results
        elif results_file:
            with open(results_file, 'r') as f:
                data = json.load(f)
                self.results = data['results']
        else:
            self.results = []
            
        # Convert to DataFrame for analysis
        self.df = self._prepare_dataframe()
        
    def _prepare_dataframe(self) -> pd.DataFrame:
        """
        Prepare DataFrame from results
        
        Returns:
            Pandas DataFrame with flattened results
        """
        if not self.results:
            return pd.DataFrame()
            
        # Extract main results
        main_data = []
        for r in self.results:
            row = {
                'persona_id': r['persona_id'],
                'treatment_group': r['treatment_group'],
                'pre_expectation': r['pre_treatment_expectation'],
                'post_expectation': r['post_treatment_expectation'],
                'expectation_change': r['expectation_change']
            }
            
            # Add persona attributes if available
            if 'persona' in r:
                for k, v in r['persona'].items():
                    row[f'persona_{k}'] = v
                    
            main_data.append(row)
            
        return pd.DataFrame(main_data)
    
    def summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics
        
        Returns:
            DataFrame with summary statistics
        """
        if self.df.empty:
            return pd.DataFrame()
            
        # Group by treatment
        stats = self.df.groupby('treatment_group').agg({
            'pre_expectation': ['count', 'mean', 'std', 'min', 'max'],
            'post_expectation': ['mean', 'std', 'min', 'max'],
            'expectation_change': ['mean', 'std', 'min', 'max']
        })
        
        # Calculate additional metrics
        treatment_effects = []
        control_mean = self.df[self.df['treatment_group'] == 'control']['expectation_change'].mean()
        
        for treatment in self.df['treatment_group'].unique():
            if treatment == 'control':
                effect = 0
            else:
                treatment_mean = self.df[self.df['treatment_group'] == treatment]['expectation_change'].mean()
                effect = treatment_mean - control_mean
                
            treatment_effects.append({
                'treatment_group': treatment,
                'treatment_effect': effect
            })
            
        effects_df = pd.DataFrame(treatment_effects).set_index('treatment_group')
        
        return stats, effects_df
    
    def run_regression_analysis(self) -> Dict:
        """
        Run regression analysis on results
        
        Returns:
            Dictionary with regression results
        """
        if self.df.empty:
            return {'error': 'No data available'}
            
        # Create dummy variables for treatment groups
        treatment_dummies = pd.get_dummies(self.df['treatment_group'], prefix='treatment', drop_first=True)
        reg_df = pd.concat([self.df, treatment_dummies], axis=1)

        # List of correct dummy variable names based on treatment groups
        treatment_dummy_names = [
            'treatment_current_inflation_target',
            'treatment_media_narrative',
            'treatment_full_policy_context',
            'treatment_policy_rate_decision'
        ]

        # Run basic regression
        try:
            # First model: treatment effects only
            formula1 = 'expectation_change ~ ' + ' + '.join(treatment_dummy_names)
            model1 = sm.OLS.from_formula(formula1, data=reg_df)
            results1 = model1.fit()

            # Second model: with demographic controls
            demographic_controls = []
            for col in reg_df.columns:
                if col.startswith('persona_') and col not in ['persona_id', 'persona_province', 'persona_region']:
                    demographic_controls.append(col)

            if demographic_controls:
                formula2 = formula1 + ' + ' + ' + '.join(demographic_controls)
                model2 = sm.OLS.from_formula(formula2, data=reg_df)
                results2 = model2.fit()
            else:
                results2 = None

            # Third model: interaction with financial literacy
            if 'persona_financial_literacy' in reg_df.columns:
                interaction_terms = []
                for t in treatment_dummy_names:
                    reg_df[f'{t}_x_literacy'] = reg_df[t] * reg_df['persona_financial_literacy']
                    interaction_terms.append(f'{t}_x_literacy')

                formula3 = formula1 + ' + persona_financial_literacy + ' + ' + '.join(interaction_terms)
                model3 = sm.OLS.from_formula(formula3, data=reg_df)
                results3 = model3.fit()
            else:
                results3 = None

            return {
                'baseline_model': results1,
                'demographic_model': results2,
                'interaction_model': results3
            }

        except Exception as e:
            print(f"Error in regression analysis: {e}")
            return {'error': str(e)}
    
    def plot_treatment_effects(self, output_file: str = None):
        """
        Plot treatment effects
        
        Args:
            output_file: Path to save the plot (if None, plot is displayed)
        """
        if self.df.empty:
            print("No data available for plotting")
            return
            
        try:
            # Calculate mean change by treatment
            treatment_means = self.df.groupby('treatment_group')['expectation_change'].agg(['mean', 'std']).reset_index()
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            # Bar plot
            bars = plt.bar(treatment_means['treatment_group'], treatment_means['mean'], 
                   yerr=treatment_means['std'], capsize=10, 
                   color=['lightgray', 'lightblue', 'lightgreen', 'lightpink', 'lightyellow'])
            
            # Add labels and title
            plt.xlabel('Treatment Group')
            plt.ylabel('Mean Change in Inflation Expectation (%)')
            plt.title(f'Treatment Effects on Inflation Expectations in {self.df["quarter"].iloc[0]}')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file)
                print(f"Plot saved to {output_file}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error in plotting: {e}")
    
    def heterogeneity_analysis(self) -> pd.DataFrame:
        """
        Analyze treatment effect heterogeneity
        
        Returns:
            DataFrame with heterogeneity analysis
        """
        if self.df.empty:
            return pd.DataFrame()
            
        # Define demographic factors for heterogeneity analysis
        factors = [
            ('persona_gender', ['Male', 'Female']),
            ('persona_urban_rural', ['Urban', 'Rural']),
            ('persona_financial_literacy', lambda x: 'High' if x > 6 else 'Low'),
            ('persona_education', lambda x: 'Higher' if x in ["Bachelor's Degree", "Master's Degree", "Doctorate"] else 'Lower'),
            ('persona_age', lambda x: 'Young' if x < 35 else 'Older')
        ]
        
        results = []
        
        # Calculate treatment effects for each demographic group
        for factor, groups in factors:
            if factor not in self.df.columns:
                continue
                
            # Create grouping variable
            if callable(groups):
                self.df[f'{factor}_group'] = self.df[factor].apply(groups)
                groups = self.df[f'{factor}_group'].unique()
            
            for group in groups:
                for treatment in self.df['treatment_group'].unique():
                    if treatment == 'control':
                        continue
                        
                    # Filter data
                    if callable(groups):
                        group_data = self.df[self.df[f'{factor}_group'] == group]
                    else:
                        group_data = self.df[self.df[factor] == group]
                    
                    # Get treatment and control for this group
                    group_treatment = group_data[group_data['treatment_group'] == treatment]
                    group_control = group_data[group_data['treatment_group'] == 'control']
                    
                    # Calculate effect
                    if len(group_treatment) > 0 and len(group_control) > 0:
                        treatment_mean = group_treatment['expectation_change'].mean()
                        control_mean = group_control['expectation_change'].mean()
                        effect = treatment_mean - control_mean
                        
                        results.append({
                            'factor': factor.replace('persona_', ''),
                            'group': group,
                            'treatment': treatment,
                            'effect': effect,
                            'treatment_n': len(group_treatment),
                            'control_n': len(group_control)
                        })
        
        return pd.DataFrame(results)


#######################
# RESULTS OUTPUT
#######################

class ResultsExporter:
    """Exports analysis results to various formats"""
    
    def __init__(self, experiment_results: Dict, output_dir: str = None):
        """
        Initialize results exporter
        
        Args:
            experiment_results: Dictionary with experiment results
            output_dir: Directory to save outputs
        """
        self.results = experiment_results
        self.output_dir = output_dir or RESULTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export_to_json(self, filename: str = None) -> str:
        """
        Export results to JSON
        
        Args:
            filename: Output filename (if None, a default name is used)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"
            
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"Results exported to {filepath}")
        return filepath
    
    def export_to_csv(self, filename: str = None) -> str:
        """
        Export results to CSV
        
        Args:
            filename: Output filename (if None, a default name is used)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.csv"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to dataframe
        rows = []
        for r in self.results['results']:
            row = {
                'persona_id': r['persona_id'],
                'treatment_group': r['treatment_group'],
                'pre_expectation': r['pre_treatment_expectation'],
                'post_expectation': r['post_treatment_expectation'],
                'expectation_change': r['expectation_change']
            }
            
            # Add persona attributes if available
            if 'persona' in r:
                for k, v in r['persona'].items():
                    row[f'persona_{k}'] = v
                    
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        
        print(f"Results exported to {filepath}")
        return filepath
    
    def generate_report(self, filename: str = None) -> str:
        """
        Generate comprehensive report with results
        
        Args:
            filename: Output filename (if None, a default name is used)
            
        Returns:
            Path to saved report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_report_{timestamp}.html"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Create analyzer
        analyzer = DataAnalyzer(results=self.results['results'])
        
        # Run analyses
        summary_stats, effects_df = analyzer.summary_statistics()
        regression_results = analyzer.run_regression_analysis()
        heterogeneity_df = analyzer.heterogeneity_analysis()
        
        # Get quarter string
        quarter_str = self.results['results'][0]['quarter'] if self.results.get('results') and self.results['results'] and 'quarter' in self.results['results'][0] else ''

        # Create plots
        plot_file = os.path.join(self.output_dir, f"treatment_effects_{quarter_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        analyzer.plot_treatment_effects(plot_file)
        
        # Generate HTML report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Bank Indonesia Inflation Expectations Experiment Report {quarter_str}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .section {{ margin-bottom: 30px; }}
                    .plot {{ text-align: center; margin: 20px 0; }}
                    pre {{ background-color: #f8f8f8; padding: 10px; overflow-x: auto; }}
                </style>
            </head>
            <body>
                <h1>Bank Indonesia Inflation Expectations Experiment Report {quarter_str}</h1>
                <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Model:</strong> {self.results['model']}</p>
                <p><strong>Sample Size:</strong> {self.results['persona_count']} personas</p>
                
                <div class="section">
                    <h2>1. Summary Statistics</h2>
                    {summary_stats.to_html()}
                    
                    <h3>Treatment Effects</h3>
                    {effects_df.to_html()}
                </div>
                
                <div class="section">
                    <h2>2. Treatment Effects Plot</h2>
                    <div class="plot">
                        <img src="{os.path.basename(plot_file)}" alt="Treatment Effects Plot">
                    </div>
                </div>
            ''')
            
            if 'error' not in regression_results:
                f.write('''
                <div class="section">
                        
                    <h2>3. Regression Analysis</h2>
                    <h3>3.1 Baseline Model (Treatment Effects Only)</h3>
                    <pre>{}</pre>
                '''.format(regression_results['baseline_model'].summary().as_html()))
                
                if regression_results['demographic_model']:
                    f.write('''
                    <h3>3.2 Model with Demographic Controls</h3>
                    <pre>{}</pre>
                    '''.format(regression_results['demographic_model'].summary().as_html()))
                    
                if regression_results['interaction_model']:
                    f.write('''
                    <h3>3.3 Model with Financial Literacy Interactions</h3>
                    <pre>{}</pre>
                    '''.format(regression_results['interaction_model'].summary().as_html()))
                    
                f.write('</div>')
            
            if not heterogeneity_df.empty:
                f.write(f'''
                <div class="section">
                    <h2>4. Heterogeneity Analysis</h2>
                    {heterogeneity_df.to_html()}
                </div>
                ''')
                
            f.write('''
                <div class="section">
                    <h2>5. Conclusion</h2>
                    <p>This report provides a comprehensive analysis of how different information treatments affect inflation expectations among Indonesian households. The experiment simulated responses from diverse demographic profiles to examine information effects and heterogeneity in updating behavior.</p>
                </div>
            </body>
            </html>
            ''')
        
        print(f"Report generated at {filepath}")
        return filepath


#######################
# MAIN EXECUTION
#######################

def main():
    """Main execution function"""
    print("Bank Indonesia Inflation Expectations Experiment")
    print("=" * 50)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run Bank Indonesia inflation expectations experiment')
    parser.add_argument('--personas_per_group', type=int, default=30, help='Number of personas per treatment group')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='LLM model to use')
    parser.add_argument('--db-path', type=str, default=None, help='Path to SQLite database file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    args = parser.parse_args()
    
    try:
        # Initialize database
        print("Initializing database...")
        db_manager = DatabaseManager(db_path=args.db_path)
        
        # Run experiment
        print(f"Running experiment with {args.personas_per_group} personas per group using {args.model}...")
        experiment_manager = ExperimentManager(db_manager)
        results = experiment_manager.run_experiment(personas_per_group=args.personas_per_group, model=args.model)
        
        # Export results
        print("Exporting results...")
        exporter = ResultsExporter(results, output_dir=args.output_dir)
        json_file = exporter.export_to_json()
        csv_file = exporter.export_to_csv()
        report_file = exporter.generate_report()
        
        print("\nExperiment completed successfully!")
        print(f"Results saved to:")
        print(f"  - JSON: {json_file}")
        print(f"  - CSV: {csv_file}")
        print(f"  - Report: {report_file}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())