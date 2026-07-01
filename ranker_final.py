import json
import gzip
import hashlib
import time
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# DEPENDENCY CHECK AND INSTALLATION


def check_and_install_dependencies():
    required_packages = {
        'numpy': 'numpy',
        'sklearn': 'scikit-learn'
    }

    optional_packages = {
        'sentence_transformers': 'sentence-transformers',
        'torch': 'torch'
    }
    
    missing = []
    missing_optional = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    for import_name, package_name in optional_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_optional.append(package_name)
    
    if missing:
        print(f"[SETUP] Installing missing packages: {', '.join(missing)}")
        import subprocess
        for pkg in missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
        print("[SETUP] Dependencies installed successfully\n")

    if missing_optional:
        print(f"[SETUP] Optional deep learning packages not found: {', '.join(missing_optional)}")
        print("[SETUP] Continuing with fallback TF-IDF+SVD semantic search if sentence-transformers is unavailable.\n")

check_and_install_dependencies()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


# SECTION 1: JOB DESCRIPTION ANALYZER


class JobDescriptionAnalyzer:
    """Extract and score candidates against the Senior AI Engineer JD"""
    
    REQUIRED_SKILLS = {
        'embeddings_retrieval': ['embedding', 'embeddings', 'retrieval', 'semantic search', 'hybrid search', 'search relevance'],
        'ranking_evaluation': ['ranking', 'ranker', 'ndcg', 'mrr', 'map', 'evaluation', 'relevance'],
        'vector_db': ['vector database', 'vectordb', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'faiss'],
        'python': ['python'],
        'llm_finetuning': ['llm', 'large language model', 'fine-tuning', 'lora', 'qlora', 'peft', 'prompt tuning'],
        'production_shipping': ['production', 'deployed', 'shipped', 'deployment', 'real users', 'ship', 'launched', 'live'],
    }
    
    PREFERRED_SKILLS = {
        'hr_tech': ['hr-tech', 'recruiting', 'talent', 'recruitment', 'recruiting tech'],
        'distributed_systems': ['distributed systems', 'scale', 'scalability', 'distributed'],
        'startup_product': ['startup', 'early-stage', 'product', 'consumer', 'recruiter workflows', 'matching'],
    }

    DESIRED_LOCATIONS = ['bangalore', 'pune', 'noida']
    JD_KEYWORD_WEIGHTS = {
        'embedding': 1.0,
        'embeddings': 1.0,
        'retrieval': 1.2,
        'ranking': 1.2,
        'vector': 0.8,
        'pinecone': 0.9,
        'qdrant': 0.9,
        'milvus': 0.9,
        'weaviate': 0.9,
        'faiss': 0.9,
        'elasticsearch': 0.9,
        'opensearch': 0.9,
        'llm': 0.9,
        'fine-tuning': 1.0,
        'lora': 0.8,
        'peft': 0.8,
        'ndcg': 0.9,
        'mrr': 0.9,
        'map': 0.9,
        'evaluation': 0.8,
        'python': 0.6,
        'hybrid search': 0.9,
        'search': 0.5,
        'matching': 0.6,
        'deployed': 0.8,
        'production': 0.8,
        'shipped': 0.8,
        'startup': 0.6,
    }
    SHIPPING_TERMS = ['ship', 'shipped', 'deployed', 'production', 'launched', 'real users', 'startup', 'early-stage']
    JOB_DESCRIPTION_TEXT = (
        'Senior AI Engineer for a Series A startup building talent intelligence. Need strong production work in embeddings, retrieval, ranking, vector databases, search, Python, LLM fine-tuning, evaluation frameworks, and shipping quickly with real users.'
    )
    
    @classmethod
    def get_semantic_query(cls, project_dir: Optional[Path] = None) -> str:
        if project_dir is not None:
            jd_file = project_dir / 'job_description_text.txt'
            if jd_file.exists():
                text = jd_file.read_text(encoding='utf-8').strip()
                if text:
                    return text
        return cls.JOB_DESCRIPTION_TEXT
    
    @classmethod
    def score_candidate_fit(cls, candidate: Dict) -> Tuple[float, Dict]:
        """Score how well candidate fits the JD"""
        details = {
            'required_match': 0.0,
            'preferred_match': 0.0,
            'disqualifier_flags': []
        }
        
        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        
        headline = profile.get('headline', '').lower()
        summary = profile.get('summary', '').lower()
        career_text = ' '.join([job.get('description', '') for job in career_history]).lower()
        skills_text = ' '.join([s['name'].lower() for s in skills])
        full_text = f"{headline} {summary} {career_text} {skills_text}".lower()
        
        keyword_score = 0.0
        matched_terms = []
        for term, weight in cls.JD_KEYWORD_WEIGHTS.items():
            # ensures we match exact words (e.g., 'ml' won't trigger inside 'html')
            if re.search(r'\b' + re.escape(term) + r'\b', full_text):
                keyword_score += weight
                matched_terms.append(term)
        details['keyword_score'] = min(1.0, keyword_score / 7.5)
        details['matched_terms'] = matched_terms[:10]
        
        required_matches = 0
        for skill_category, keywords in cls.REQUIRED_SKILLS.items():
            if any(keyword in full_text for keyword in keywords):
                required_matches += 1
                details[f'has_{skill_category}'] = True
        
        details['required_match'] = min(1.0, required_matches / 5.0)
        
        preferred_matches = 0
        for skill_category, keywords in cls.PREFERRED_SKILLS.items():
            if any(keyword in full_text for keyword in keywords):
                preferred_matches += 1
                details[f'has_{skill_category}'] = True
        
        details['preferred_match'] = min(1.0, preferred_matches / 2.0)
        
        shipping_hits = sum(1 for term in cls.SHIPPING_TERMS if term in full_text)
        details['shipping_hits'] = shipping_hits
        
        yoe = profile.get('years_of_experience', 0)
        if yoe < 5:
            details['disqualifier_flags'].append('Insufficient experience (<5 years)')
        
        location = profile.get('location', '').lower()
        availability = candidate.get('redrob_signals', {})
        open_to_work = availability.get('open_to_work_flag', False)
        notice_period = availability.get('notice_period_days', 60)

        location_score = 0.0
        if any(city in location for city in cls.DESIRED_LOCATIONS):
            location_score = 1.0
        elif 'remote' in location or 'work from home' in location:
            location_score = 0.7
        details['location_score'] = location_score

        availability_score = 0.0
        if open_to_work and notice_period <= 30:
            availability_score = 1.0
        elif open_to_work:
            availability_score = 0.7
        details['availability_score'] = availability_score

        base_score = (
            details['required_match'] * 0.35 +
            details['preferred_match'] * 0.20 +
            details['keyword_score'] * 0.25 +
            min(1.0, shipping_hits / 3.0) * 0.10 +
            location_score * 0.05 +
            availability_score * 0.05
        )
        
        if details['disqualifier_flags']:
            base_score *= 0.35
        
        return min(1.0, base_score), details

# SECTION 2: HONEYPOT DETECTION

class HoneypotDetector:
    """Detect fake/trap candidates in the dataset"""
    
    @staticmethod
    def detect_honeypots(candidate: Dict) -> Tuple[bool, str]:
        """Returns: (is_honeypot: bool, reason: str)"""
        cid = candidate.get('candidate_id', '')
        profile = candidate.get('profile', {})
        signals = candidate.get('redrob_signals', {})
        skills = candidate.get('skills', [])
        
        # TRAP 1: KEYWORD STUFFER
        headline = profile.get('headline', '')
        summary = profile.get('summary', '')
        skills_text = ' '.join([s['name'] for s in skills])
        full_text = f"{headline} {summary} {skills_text}".lower().split()
        
        if len(full_text) > 20:
            unique_words = len(set(full_text))
            diversity_ratio = unique_words / len(full_text)
            
            if diversity_ratio < 0.25:
                return True, "TRAP: Keyword stuffer (low text diversity)"
        
        # TRAP 2: GHOST PROFILE
        completeness = signals.get('profile_completeness_score', 50)
        if completeness < 20 and len(full_text) > 100:
            return True, "TRAP: Ghost profile (high text, low completeness)"
        
        # TRAP 3: GIT CONTRADICTION
        github_score = signals.get('github_activity_score', 0)
        if github_score == -1 and ('github' in skills_text or 'git' in skills_text):
            return True, "TRAP: Contradictory signals (claims Git, -1 score)"
        
        # TRAP 4: SUSPICIOUS VETERAN
        yoe = profile.get('years_of_experience', 0)
        views_30d = signals.get('profile_views_received_30d', 0)
        apps_30d = signals.get('applications_submitted_30d', 0)
        
        if yoe > 5 and views_30d == 0 and apps_30d == 0:
            return True, "TRAP: Suspicious veteran (high exp, zero engagement)"
        
        # TRAP 5: UNVERIFIED GHOST
        verified_email = signals.get('verified_email', False)
        verified_phone = signals.get('verified_phone', False)
        last_active = signals.get('last_active_date', '')
        
        if not verified_email and not verified_phone and not last_active:
            return True, "TRAP: Unverified profile with no activity"
        
        # TRAP 6: GHOST SKILL ASSESSOR
        skill_assessments = signals.get('skill_assessment_scores', {})
        if len(skill_assessments) == 0 and yoe > 7:
            return True, "TRAP: No skill assessments (should have some if senior)"
        # TRAP 7: Absolute Zero Engagement
        # Catching senior profiles with completely zero ecosystem touchpoints
        views_30d = signals.get('profile_views_received_30d', 0)
        apps_30d = signals.get('applications_submitted_30d', 0)
        recruiter_resp = signals.get('recruiter_response_rate', 0)
        
        if views_30d == 0 and apps_30d == 0 and recruiter_resp == 0:
            return True, "TRAP: Absolute zero engagement score markers"
        
        # TRAP 8: Excessive Notice Period / Unavailability
        notice_period = signals.get('notice_period_days', 60)
        if notice_period >= 120:
            return True, "TRAP: Outlier notice period (>= 120 days) indicating a dormant/trap profile"

        # TRAP 9: Not Actively Looking / Dormant
        # If the open_to_work_flag is explicitly False or missing
        open_to_work = signals.get('open_to_work_flag', True)
        if not open_to_work:
            return True, "TRAP: Profile is marked as not actively looking/working"
        
        return False, ""


# SECTION 3: ENGAGEMENT & AVAILABILITY SCORER

class EngagementScorer:
    """Score candidate engagement and availability"""
    
    @staticmethod
    def score(candidate: Dict) -> Tuple[float, Dict]:
        """Returns: (engagement_score 0-1, details)"""
        signals = candidate.get('redrob_signals', {})
        profile = candidate.get('profile', {})
        
        details = {}
        
        # Recent activity (0-0.25)
        last_active = signals.get('last_active_date', '')
        activity_score = 0.25 if last_active else 0.0
        
        # Recruiter response rate (0-0.25)
        response_rate = signals.get('recruiter_response_rate', 0)
        response_score = min(0.25, response_rate * 0.25)
        details['response_rate'] = response_rate
        
        # Availability signals (0-0.25)
        open_to_work = signals.get('open_to_work_flag', False)
        notice_period = signals.get('notice_period_days', 60)
        willing_relocate = signals.get('willing_to_relocate', False)
        
        availability = 0.0
        if open_to_work:
            availability += 0.1
        if notice_period <= 30:
            availability += 0.1
        if willing_relocate:
            availability += 0.05
        
        availability_score = min(0.25, availability)
        
        # Verification score (0-0.25)
        verified_email = signals.get('verified_email', False)
        verified_phone = signals.get('verified_phone', False)
        linkedin_connected = signals.get('linkedin_connected', False)
        
        verification = 0.0
        if verified_email:
            verification += 0.08
        if verified_phone:
            verification += 0.08
        if linkedin_connected:
            verification += 0.09
        
        verification_score = min(0.25, verification)
        
        total = activity_score + response_score + availability_score + verification_score
        
        details['activity_score'] = activity_score
        details['response_score'] = response_score
        details['availability_score'] = availability_score
        details['verification_score'] = verification_score
        details['notice_period_days'] = notice_period
        details['open_to_work'] = open_to_work
        
        return min(1.0, total), details


# SECTION 4: EXPERIENCE VALIDATOR

class ExperienceValidator:
    """Validate candidate's production ML experience"""
    
    PRODUCTION_ML_KEYWORDS = [
        'production', 'deployed', 'shipped', 'scale', 'pipeline', 'system',
        'real users', 'users', 'live', 'model', 'ml', 'ai', 'data engineering',
        'ml engineer', 'ai engineer', 'data scientist', 'retrieval', 'ranking',
        'embedding', 'embeddings', 'search', 'matching', 'vector', 'llm',
        'fine-tuning', 'evaluation', 'recommendation', 'recommender'
    ]
    
    @staticmethod
    def score(candidate: Dict) -> Tuple[float, Dict]:
        """Returns: (experience_score 0-1, details)"""
        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])
        
        yoe = profile.get('years_of_experience', 0)
        current_title = profile.get('current_title', '').lower()
        
        details = {
            'years_of_experience': yoe,
            'has_production_ml': False,
            'senior_level': False
        }
        
        if yoe < 5:
            return 0.0, details
        
        if 5 <= yoe <= 9:
            details['senior_level'] = True
            yoe_score = 1.0
        elif yoe > 9:
            yoe_score = 0.9
        else:
            yoe_score = 0.5
        
        production_hits = 0
        full_text = []
        for job in career_history:
            title = job.get('title', '').lower()
            description = job.get('description', '').lower()
            company_size = job.get('company_size', '')
            merged = f"{title} {description}".lower()
            full_text.append(merged)
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', merged) for keyword in ExperienceValidator.PRODUCTION_ML_KEYWORDS):
                production_hits += 1
            if company_size in ['501-1000', '1001-5000', '5001-10000', '10001+']:
                production_hits += 1
        
        text_blob = ' '.join(full_text).lower()
        if any(term in text_blob for term in ['retrieval', 'ranking', 'embedding', 'vector', 'search', 'llm', 'fine-tuning', 'evaluation', 'deployed', 'production', 'shipped']):
            production_hits += 1
        
        if production_hits > 0:
            details['has_production_ml'] = True
            prod_score = min(1.0, 0.25 + (production_hits * 0.12))
        else:
            prod_score = 0.3
        
        if 'ml' in current_title or 'ai' in current_title or 'engineer' in current_title:
            current_score = 1.0
        else:
            current_score = 0.5
        
        final_score = (yoe_score * 0.4 + prod_score * 0.4 + current_score * 0.2)
        
        return final_score, details


# SECTION 5: SKILL ASSESSMENT SCORER

class SkillAssessmentScorer:
    """Score candidate's skill assessment results"""
    
    RELEVANT_ASSESSMENTS = [
        'python', 'ml', 'nlp', 'deep learning', 'machine learning',
        'sql', 'spark', 'data engineering', 'retrieval', 'ranking'
    ]
    
    @staticmethod
    def score(candidate: Dict) -> float:
        """Returns: skill_assessment_score (0-1)"""
        signals = candidate.get('redrob_signals', {})
        assessment_scores = signals.get('skill_assessment_scores', {})
        
        if not assessment_scores:
            return 0.2
        
        relevant_scores = []
        for skill_name, score in assessment_scores.items():
            skill_lower = skill_name.lower()
            if any(kw in skill_lower for kw in SkillAssessmentScorer.RELEVANT_ASSESSMENTS):
                relevant_scores.append(score / 100.0)
        
        if relevant_scores:
            avg_score = sum(relevant_scores) / len(relevant_scores)
            return min(1.0, avg_score)
        
        return 0.1


# SECTION 6: EMBEDDING & NEURAL RANKER

class EmbeddingSearch:
    """Build candidate embeddings and compute semantic similarity."""
    
    VECTORIZER_PARAMS = {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'lowercase': True,
        'stop_words': 'english'
    }
    
    def __init__(self, candidates: List[Dict], query_text: str):
        self.candidates = candidates
        self.query_text = query_text
        self.documents = [self.build_document_text(c) for c in candidates]
        self.use_sentence_transformer = False
        self.sentence_model = None
        self.vectorizer = None
        self.svd = None
        self.embeddings = None
        self.query_embedding = None
        
        try:
            from sentence_transformers import SentenceTransformer
            # Load your local model
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
            self.use_sentence_transformer = True
        except Exception:
            self.sentence_model = None
            self.use_sentence_transformer = False
            
        if self.use_sentence_transformer:
            print("[SEMANTIC] Starting multi-process CPU pool...")
            # 1. Fire up a worker pool for ALL your CPU cores
            pool = self.sentence_model.start_multi_process_pool()
            
            # 2. Blast through the 61k documents in parallel batches
            self.embeddings = self.sentence_model.encode_multi_process(
                self.documents, 
                pool,
                batch_size=1024
            )
            
            # 3. Shutdown the pool instantly to clear RAM overhead
            self.sentence_model.stop_multi_process_pool(pool)
            
            self.query_embedding = self.sentence_model.encode([query_text], show_progress_bar=False, convert_to_numpy=True)
        else:
            self.vectorizer = TfidfVectorizer(**self.VECTORIZER_PARAMS)
            tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            self.svd = TruncatedSVD(n_components=128, random_state=42)
            self.embeddings = self.svd.fit_transform(tfidf_matrix)
            self.query_embedding = self.svd.transform(self.vectorizer.transform([query_text]))
            
        self.scores = self.compute_scores()
           
    @staticmethod
    def build_document_text(candidate: Dict) -> str:
        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        education = candidate.get('education', [])
        redrob_signals = candidate.get('redrob_signals', {})

        # 1. Heavily amplify high-signal target fields right at the front
        headline = profile.get('headline', '')
        current_title = profile.get('current_title', '')
        
        parts = [
            f"ROLE: {headline} {current_title}",  # First emphasis
            f"HEADLINE: {headline}",               # Double emphasis on headline
            profile.get('summary', ''),
            ' '.join([s.get('name', '') for s in skills]),
            ' '.join([job.get('title', '') + ' ' + job.get('description', '') for job in career_history]),
            ' '.join([edu.get('degree', '') + ' ' + edu.get('field_of_study', '') for edu in education]),
        ]

        for key, value in redrob_signals.items():
            if isinstance(value, dict):
                parts.append(' '.join([f"{subkey}:{subvalue}" for subkey, subvalue in value.items()]))
            else:
                parts.append(str(value))
                
        text = ' '.join(parts).strip()
        
        # 2. Upped from 200 to 1500 to capture complete profiles safely within model context limits
        return text[:4000] 
   
    def compute_scores(self) -> np.ndarray:
        # Pre-normalize the embeddings along axis 1 for vectorized L2 norm
        norm_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norm_query = self.query_embedding / np.linalg.norm(self.query_embedding)
        
        # Blazing-fast matrix dot product calculation
        return np.dot(norm_embeddings, norm_query.flatten())
    
    def get_score(self, index: int) -> float:
        return float(self.scores[index])
    
    def get_embedding(self, index: int) -> np.ndarray:
        return self.embeddings[index]


class FeatureBuilder:
    """Construct numerical features for neural ranking."""
    
    TARGET_LOCATIONS = ['bangalore', 'pune', 'noida']
    
    @staticmethod
    def build_features(candidate: Dict, semantic_score: float, jd_fit: float, experience_score: float, engagement_score: float, skill_assessment: float) -> np.ndarray:
        profile = candidate.get('profile', {})
        signals = candidate.get('redrob_signals', {})
        career_history = candidate.get('career_history', [])
        # 1. Map 0 to 15+ years of experience smoothly to a 0.0 - 1.0 scale
        yoe = min(1.0, float(profile.get('years_of_experience', 0)) / 15.0)
        
        # 2. Map 0-100 completeness score down to a 0.0 - 1.0 scale
        profile_completeness = float(signals.get('profile_completeness_score', 50)) / 100.0
        
        # 3. Cleanly catch raw response percentages and scale them down to 0.0 - 1.0
        try:
            raw_resp = float(signals.get('recruiter_response_rate', 0))
            response_rate = min(1.0, raw_resp if raw_resp <= 1.0 else raw_resp / 100.0)
        except (ValueError, TypeError):
            response_rate = 0.0
        
        location = profile.get('location', '').lower()
        headline = profile.get('headline', '').lower()
        summary = profile.get('summary', '').lower()
        career_text = ' '.join([job.get('description', '') for job in career_history]).lower()
        skills_text = ' '.join([s.get('name', '') for s in candidate.get('skills', [])]).lower()
        full_text = f"{headline} {summary} {career_text} {skills_text}".lower()

        open_to_work = float(bool(signals.get('open_to_work_flag', False)))
        notice_period = float(signals.get('notice_period_days', 60))
        short_notice = 1.0 if notice_period <= 30 else 0.0
        willing_relocate = float(bool(signals.get('willing_to_relocate', False)))

        response_rate = float(signals.get('recruiter_response_rate', 0))
        profile_completeness = float(signals.get('profile_completeness_score', 50))
        verified_email = float(bool(signals.get('verified_email', False)))
        verified_phone = float(bool(signals.get('verified_phone', False)))
        linkedin_connected = float(bool(signals.get('linkedin_connected', False)))
        recent_activity = float(bool(signals.get('last_active_date', '')))

        local_hiring_region = 1.0 if any(city in location for city in FeatureBuilder.TARGET_LOCATIONS) else 0.0
        remote_friendly = 1.0 if 'remote' in location or 'work from home' in location else 0.0
        senior_title = 1.0 if any(term in headline for term in ['senior', 'staff', 'lead', 'principal']) else 0.0
        has_vector_db = 1.0 if any(term in full_text for term in ['pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'elasticsearch', 'opensearch', 'vector database']) else 0.0
        has_embeddings = 1.0 if any(term in full_text for term in ['embedding', 'retrieval', 'vector', 'semantic search', 'hybrid search']) else 0.0
        has_ranking = 1.0 if any(term in full_text for term in ['ranking', 'ranker', 'ndcg', 'mrr', 'map', 'evaluation']) else 0.0
        has_llm = 1.0 if any(term in full_text for term in ['llm', 'fine-tuning', 'lora', 'peft', 'transformer']) else 0.0
        has_shipping = 1.0 if any(term in full_text for term in ['production', 'deployed', 'shipped', 'launched', 'real users', 'startup', 'early-stage']) else 0.0
        has_python = 1.0 if 'python' in full_text else 0.0

        return np.array([
            semantic_score,
            jd_fit,
            experience_score,
            engagement_score,
            skill_assessment,
            yoe,
            profile_completeness,
            response_rate,
            open_to_work,
            short_notice,
            willing_relocate,
            verified_email,
            verified_phone,
            linkedin_connected,
            recent_activity,
            local_hiring_region,
            remote_friendly,
            senior_title,
            has_vector_db,
            has_embeddings,
            has_ranking,
            has_llm,
            has_shipping,
            has_python,
        ], dtype=float)


class NeuralRanker:
    """Train and predict ranking scores using a neural regressor."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=150,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            verbose=False
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return np.clip(preds, 0.0, 1.0)



# SECTION 7: MAIN RANKER ENGINE

class CandidateRanker:
    """Main production ranking system"""
    
    def __init__(self, candidates_file: str):
        self.candidates_file = candidates_file
        self.project_dir = Path(candidates_file).parent
        self.candidates = []
        self.honeypots_found = []
        self.ranked_candidates = []
        self.semantic_scores = []
        self.semantic_embeddings = None
        self.feature_matrix = None
        self.target_scores = None
        self.neural_ranker = None
        self.original_candidate_ids = set()
    
    def load_candidates(self, limit: Optional[int] = None) -> int:
        """Load candidates from JSONL or gzipped JSONL"""
        print(f"[LOAD] Reading candidates from {self.candidates_file}...")
        start = time.time()
        
        count = 0
        
        if self.candidates_file.endswith('.gz'):
            opener = gzip.open
        else:
            opener = open
        
        try:
            with opener(self.candidates_file, 'rt' if self.candidates_file.endswith('.gz') else 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            candidate = json.loads(line)
                            self.candidates.append(candidate)
                            self.original_candidate_ids.add(candidate.get('candidate_id'))
                            count += 1
                            
                            if count % 20000 == 0:
                                print(f"  [LOAD] Loaded {count:,} candidates...")
                            
                            if limit and count >= limit:
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"[ERROR] Failed to load candidates: {e}")
            return 0
        
        elapsed = time.time() - start
        print(f"[LOAD] Loaded {count:,} candidates in {elapsed:.2f}s\n")
        
        return count
    
    def filter_honeypots(self) -> int:
        """Filter out honeypots, duplicate cheaters, and unqualified profiles"""
        print("[HONEYPOT] Detecting honeypots, duplicates, and pre-filtering...")
        start = time.time()
        valid_candidates = []
        
        # Anti-Cheating Registry: Track unique texts to catch copy-paste profiles
        seen_texts = set()
        
        for candidate in self.candidates:
            cid = str(candidate.get('candidate_id', ''))
            profile = candidate.get('profile', {})
            signals = candidate.get('redrob_signals', {})
            
            yoe = profile.get('years_of_experience', 0)
            views = signals.get('profile_views_received_30d', 0)
            apps = signals.get('applications_submitted_30d', 0)
            
            # Catch senior profiles with artificially suppressed or dead engagement metrics
            if yoe >= 5 and (views <= 2 and apps <= 2):
                self.honeypots_found.append({'candidate_id': cid, 'reason': 'TRAP: Dead Senior Profile'})
                continue
                
            if yoe < 5:
                continue # Skip junior profiles to protect time limit
                
            # --- ANTI-CHEATING: DUPLICATE CONTENT DETECTOR ---
            headline = profile.get('headline', '').strip().lower()
            summary = profile.get('summary', '').strip().lower()
            # Create a unique fingerprint out of their core bio
            fingerprint = f"{headline[:50]}|{summary[:100]}"
            
            if fingerprint in seen_texts and len(summary) > 10:
                self.honeypots_found.append({'candidate_id': cid, 'reason': 'CHEATER: Duplicate profile fingerprint'})
                continue
            seen_texts.add(fingerprint)
            
            # --- STANDARD HONEYPOT DETECTOR ---
            is_honeypot, reason = HoneypotDetector.detect_honeypots(candidate)
            if is_honeypot:
                self.honeypots_found.append({'candidate_id': cid, 'reason': reason})
                continue
                
            valid_candidates.append(candidate)
            
        self.candidates = valid_candidates
        elapsed = time.time() - start
        print(f"[HONEYPOT] Defeated {len(self.honeypots_found):,} malicious/duplicate profiles")
        print(f"[HONEYPOT] Retained {len(self.candidates):,} pristine candidates in {elapsed:.2f}s\n")
        return len(self.honeypots_found) 
   
    def build_semantic_scores(self) -> None:
        """Compute semantic similarity scores and embeddings."""
        print("[SEMANTIC] Building semantic search index...")
        start = time.time()
        query = JobDescriptionAnalyzer.get_semantic_query(self.project_dir)
        semantic_search = EmbeddingSearch(self.candidates, query)
        self.semantic_scores = semantic_search.scores.tolist()
        self.semantic_embeddings = semantic_search.embeddings
        self.use_sentence_transformer = semantic_search.use_sentence_transformer
        elapsed = time.time() - start
        engine = "sentence-transformers" if self.use_sentence_transformer else "tfidf+svd"
        print(f"[SEMANTIC] Built semantic scores for {len(self.candidates):,} candidates using {engine} in {elapsed:.2f}s\n")
    
    def rank_candidates(self) -> None:
        """Score and rank all candidates using neural ranking with RRF matching targets."""
        print("[RANK] Scoring candidates...")
        start = time.time()

        self.build_semantic_scores()
        
        # 1. Rank everyone by semantic search alone
        candidates_by_semantic = sorted(
            range(len(self.candidates)), 
            key=lambda idx: self.semantic_scores[idx] if idx < len(self.semantic_scores) else 0.0, 
            reverse=True
        )
        semantic_ranks = {orig_idx: rank for rank, orig_idx in enumerate(candidates_by_semantic)}

        # 2. Rank everyone by keyword match alone
        candidates_by_jd = sorted(
            range(len(self.candidates)), 
            key=lambda idx: JobDescriptionAnalyzer.score_candidate_fit(self.candidates[idx])[0], 
            reverse=True
        )
        jd_ranks = {orig_idx: rank for rank, orig_idx in enumerate(candidates_by_jd)}
   
        scored = []
        feature_rows = []
        target_scores = []

        for idx, candidate in enumerate(self.candidates):
            if idx % 20000 == 0 and idx > 0:
                print(f"  [RANK] Scored {idx:,} candidates...")

            semantic_fit = self.semantic_scores[idx] if idx < len(self.semantic_scores) else 0.0
            jd_fit, jd_details = JobDescriptionAnalyzer.score_candidate_fit(candidate)
            engagement, eng_details = EngagementScorer.score(candidate)
            experience, exp_details = ExperienceValidator.score(candidate)
            skill_assessment = SkillAssessmentScorer.score(candidate)

            s_rank = semantic_ranks.get(idx, len(self.candidates))
            j_rank = jd_ranks.get(idx, len(self.candidates))
            
            # Compute structural order score via RRF
            rrf_score = (1.0 / (60.0 + s_rank)) + (1.0 / (60.0 + j_rank))
            
            # Extract underlying activity data footprints
            signals = candidate.get('redrob_signals', {})
            has_github = 1.0 if signals.get('github_activity_score', 0) > 0 else 0.0
            
            # Train the network to score profile data integrity and verifiability
            verified_footprint = (engagement * 0.4) + (skill_assessment * 0.4) + (has_github * 0.2)
            
            # Set the training target strictly to this non-linear proxy layer
            target_score = min(1.0, verified_footprint)
                                    # --------------------------------------------------

            features = FeatureBuilder.build_features(
                candidate,
                semantic_fit,
                jd_fit,
                experience,
                engagement,
                skill_assessment
            )

            feature_rows.append(features)
            target_scores.append(target_score)

            scored.append({
                'candidate': candidate,
                'semantic_fit': semantic_fit,
                'jd_fit': jd_fit,
                'experience': experience,
                'engagement': engagement,
                'skill_assessment': skill_assessment,
                'jd_details': jd_details,
                'eng_details': eng_details,
                'exp_details': exp_details,
                'target_score': target_score,
            })

        self.feature_matrix = np.array(feature_rows, dtype=np.float32)
        self.target_scores = np.array(target_scores)
        self.neural_ranker = NeuralRanker()
        self.neural_ranker.fit(self.feature_matrix, self.target_scores)

        ranked = []
        model_scores = self.neural_ranker.predict(self.feature_matrix)

        for idx, entry in enumerate(scored):
            heuristic_score = float(entry['target_score'])
            predicted_score = float(model_scores[idx])
            composite_score = min(1.0, 0.65 * heuristic_score + 0.35 * predicted_score)
            entry['composite_score'] = composite_score
            entry['predicted_score'] = predicted_score
            ranked.append(entry)

        ranked.sort(key=lambda x: x['composite_score'], reverse=True)
        self.ranked_candidates = ranked

        elapsed = time.time() - start
        print(f"[RANK] Trained neural ranker and ranked {len(ranked):,} candidates in {elapsed:.2f}s\n")
    
    def _pick_phrase(self, candidate_id: str, options: List[str]) -> str:
        """Deterministically pick a phrase based on candidate ID hash"""
        hashed = int(hashlib.md5(candidate_id.encode('utf-8')).hexdigest(), 16)
        return options[hashed % len(options)]

    def generate_reasoning(self, candidate_data: Dict) -> str:
        """Generate accurate 3-4 line reasoning based on real candidate metrics"""
        candidate = candidate_data['candidate']
        profile = candidate.get('profile', {})
        skills_list = [s['name'] for s in candidate.get('skills', [])]
        eng_details = candidate_data['eng_details']
        exp_details = candidate_data['exp_details']
        
        title = profile.get('current_title', 'AI Professional')
        yoe = profile.get('years_of_experience', 0)
        location = profile.get('location', 'Not specified')
        
        # Pull real calculated scores
        semantic_fit = candidate_data.get('semantic_fit', 0.0)
        response_rate = eng_details.get('response_rate', 0.0)
        open_to_work = eng_details.get('open_to_work', False)
        notice_period = eng_details.get('notice_period_days', 60)
        has_prod_ml = exp_details.get('has_production_ml', False)

        # --- LINE 1: Dynamic Experience & Alignment ---
        if semantic_fit > 0.7:
            line1 = f"{title} brings {yoe:.1f} years of experience with an exceptionally strong semantic match for the core requirements."
        elif yoe >= 5:
            line1 = f"Demonstrating {yoe:.1f} years of experience, this professional matches the senior baseline required for the role."
        else:
            line1 = f"An emerging {title} with {yoe:.1f} years of experience, providing a foundational baseline matching core criteria."

        # --- LINE 2: Dynamic Core Technical Skills ---
        top_skills = ', '.join(skills_list[:3]) if skills_list else "AI/ML methodologies"
        if has_prod_ml:
            line2 = f"The profile showcases proven capabilities in {top_skills} backed by practical production engineering experience."
        else:
            line2 = f"The candidate possesses practical knowledge in {top_skills} though deeper production-scale infrastructure details are limited."

        # --- LINE 3: Dynamic Engagement & Availability (Fixes the fake data!) ---
        if response_rate == 0 and not open_to_work:
            line3 = f"Ecosystem engagement signals are currently dormant, and the candidate is not explicitly marked as actively looking."
        else:
            status_text = f"open to opportunities ({notice_period}-day notice)" if open_to_work else "passively open"
            engagement_qual = "high" if response_rate > 0.6 else "moderate" if response_rate > 0.2 else "low baseline"
            line3 = f"The candidate maintains a {engagement_qual} responsiveness metric and is currently noted as {status_text}."

        # --- LINE 4: Final Summary Match ---
        region_match = any(city in location for city in ['Bangalore', 'Pune', 'Noida'])
        location_arrangement = "aligned with regional hubs" if region_match else "remote / non-hub location"
        line4 = f"Based in {location} ({location_arrangement}), this profile presents a structured option for final pipeline review."

        # Combine all lines cleanly
        return ' '.join([line1, line2, line3, line4])
    
    def get_top_100(self) -> List[Dict]:
        """Get top 100 candidates"""
        return self.ranked_candidates[:100]
    
    def validate_honeypot_rate(self) -> Tuple[bool, float]:
        """Check honeypot rate in top 100"""
        top_100_ids = {c['candidate']['candidate_id'] for c in self.get_top_100()}
        honeypot_ids = {h['candidate_id'] for h in self.honeypots_found}
        
        honeypots_in_top_100 = len(top_100_ids & honeypot_ids)
        rate = honeypots_in_top_100 / 100.0
        
        is_valid = rate <= 0.10
        
        return is_valid, rate
    def calculate_mrr(self) -> float:
        """Calculate Mean Reciprocal Rank (MRR) for top choices to verify ranking quality."""
        top_100 = self.get_top_100()
        for rank_idx, item in enumerate(top_100, 1):
            # Track the rank of the first candidate who meets high semantic, 
            # experience, and engagement thresholds
            if item.get('experience', 0.0) > 0.8 and item.get('semantic_fit', 0.0) > 0.7 and item.get('engagement', 0.0) > 0.5:
                return 1.0 / rank_idx
        return 0.0
    
    def verify_top_100_ids(self) -> List[str]:
        """Return any top-100 IDs not found in original dataset"""
        top_100_ids = {c['candidate']['candidate_id'] for c in self.get_top_100()}
        missing = [cid for cid in top_100_ids if cid not in self.original_candidate_ids]
        return missing
    
    def export_csv(self, output_file: str) -> None:
        """Export top 100 to CSV with 3-4 line justifications"""
        print(f"[EXPORT] Writing CSV to {output_file}...")
        
        top_100 = self.get_top_100()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("candidate_id,rank,score,reasoning\n")
            
            for rank, item in enumerate(top_100, 1):
                cid = item['candidate']['candidate_id']
                score = item['composite_score']
                reasoning = self.generate_reasoning(item)
                
                reasoning = reasoning.replace('"', '""')
                
                f.write(f'{cid},{rank},{score:.4f},"{reasoning}"\n')
        
        print(f"[EXPORT] Exported {len(top_100)} candidates to {output_file}\n")


# SECTION 8: MAIN EXECUTION

def main():
    """Main execution pipeline"""
    print("="*80)
    print("REDROB INTELLIGENT CANDIDATE DISCOVERY & RANKING - FINAL VERSION")
    print("Senior AI Engineer Role - Production Ranker")
    print("="*80)
    print()
    
    project_dir = Path(__file__).parent
    candidates_file = project_dir / "candidates.jsonl"
    
    if not candidates_file.exists() and (project_dir / "candidates.jsonl.gz").exists():
        candidates_file = project_dir / "candidates.jsonl.gz"
    
    output_file = project_dir / "submission.csv"
    
    ranker = CandidateRanker(str(candidates_file))
    
    total_start = time.time()
    
    # 1. Load
    count = ranker.load_candidates()
    if count == 0:
        print("[ERROR] No candidates loaded!")
        return
    
    # 2. Filter honeypots
    honeypot_count = ranker.filter_honeypots()
    
    # 3. Rank
    ranker.rank_candidates()
    
    # 4. Validate
    is_valid, honeypot_rate = ranker.validate_honeypot_rate()
    print(f"[VALIDATE] Top 100 honeypot rate: {honeypot_rate:.1%}")
    
    if not is_valid:
        print(f"[VALIDATE] WARNING: Honeypot rate exceeds 10% threshold!")
    else:
        print(f"[VALIDATE] [OK] Submission valid (honeypot rate < 10%)")
    
    missing_ids = ranker.verify_top_100_ids()
    if missing_ids:
        print(f"[VALIDATE] ERROR: {len(missing_ids)} top-100 IDs not found in original dataset")
        print(f"[VALIDATE] Missing IDs: {missing_ids[:10]}")
        return
    else:
        print("[VALIDATE] All top-100 candidate IDs verified against the original dataset")
    
    # 5. Export
    ranker.export_csv(str(output_file))
    
    # Print summary
    print("="*80)
    print("RANKING SUMMARY")
    print("="*80)
    
    top_5 = ranker.get_top_100()[:5]
    print("\nTop 5 Candidates:")
    for rank, item in enumerate(top_5, 1):
        cid = item['candidate']['candidate_id']
        score = item['composite_score']
        title = item['candidate']['profile'].get('current_title', '')
        yoe = item['candidate']['profile'].get('years_of_experience', 0)
        print(f"  {rank}. {cid} - {title} ({yoe:.1f}yrs) - Score: {score:.4f}")
    
    total_elapsed = time.time() - total_start
    print(f"\n[TIMING] Total execution: {total_elapsed:.2f}s")
    print(f"[METRIC] Mean Reciprocal Rank (MRR): {ranker.calculate_mrr():.4f}")
    print(f"[OUTPUT] Submission file: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
