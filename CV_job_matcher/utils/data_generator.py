"""
Sentetik Veri Oluşturma Modülü
CV Odaklı Akıllı İş Bulma Platformu için sentetik iş ilanları ve CV verileri üretir.
"""

import pandas as pd
import random
from typing import List, Dict
import json

class SyntheticDataGenerator:
    """Sentetik iş ilanı ve CV verileri oluşturan sınıf"""
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Tekrarlanabilirlik için random seed
        """
        random.seed(seed)
        self._initialize_data_pools()
    
    def _initialize_data_pools(self):
        """Veri havuzlarını başlatır"""
        
        # Sektörler
        self.sectors = [
            "Yazılım Geliştirme",
            "Veri Bilimi",
            "Yapay Zeka",
            "Siber Güvenlik",
            "Finans Teknolojileri",
            "E-Ticaret",
            "Sağlık Teknolojileri",
            "Oyun Geliştirme",
            "Bulut Bilişim",
            "IoT ve Gömülü Sistemler"
        ]
        
        # İş pozisyonları (sektörlere göre)
        self.job_titles = {
            "Yazılım Geliştirme": [
                "Full Stack Developer", "Backend Developer", "Frontend Developer",
                "Mobile Developer", "DevOps Engineer", "Software Architect"
            ],
            "Veri Bilimi": [
                "Data Scientist", "Data Analyst", "Business Intelligence Analyst",
                "Data Engineer", "ML Ops Engineer", "Analytics Manager"
            ],
            "Yapay Zeka": [
                "AI/ML Engineer", "NLP Engineer", "Computer Vision Engineer",
                "Research Scientist", "Deep Learning Engineer", "AI Product Manager"
            ],
            "Siber Güvenlik": [
                "Security Engineer", "Penetration Tester", "Security Analyst",
                "SOC Analyst", "CISO", "Security Architect"
            ],
            "Finans Teknolojileri": [
                "Fintech Developer", "Quantitative Analyst", "Blockchain Developer",
                "Payment Systems Engineer", "Risk Analyst", "Algorithmic Trader"
            ],
            "E-Ticaret": [
                "E-Commerce Manager", "Digital Marketing Specialist", "Product Manager",
                "UX Designer", "Growth Hacker", "Marketplace Developer"
            ],
            "Sağlık Teknolojileri": [
                "Health Informatics Specialist", "Medical Software Engineer",
                "Bioinformatics Scientist", "Telemedicine Developer", "Clinical Data Analyst"
            ],
            "Oyun Geliştirme": [
                "Game Developer", "Unity Developer", "Unreal Engine Developer",
                "Game Designer", "Graphics Programmer", "Technical Artist"
            ],
            "Bulut Bilişim": [
                "Cloud Engineer", "AWS Solutions Architect", "Azure Administrator",
                "Cloud Security Specialist", "Kubernetes Engineer", "Site Reliability Engineer"
            ],
            "IoT ve Gömülü Sistemler": [
                "IoT Engineer", "Embedded Systems Developer", "Firmware Engineer",
                "Hardware Engineer", "Robotics Engineer", "FPGA Developer"
            ]
        }
        
        # Beceriler (sektörlere göre)
        self.skills = {
            "Yazılım Geliştirme": [
                "Python", "JavaScript", "Java", "C++", "Go", "React", "Node.js",
                "Docker", "Kubernetes", "Git", "REST API", "GraphQL", "Microservices",
                "CI/CD", "Agile", "PostgreSQL", "MongoDB", "Redis"
            ],
            "Veri Bilimi": [
                "Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
                "PyTorch", "Tableau", "Power BI", "Apache Spark", "Hadoop",
                "Statistical Analysis", "Data Visualization", "ETL", "A/B Testing"
            ],
            "Yapay Zeka": [
                "Python", "TensorFlow", "PyTorch", "Keras", "Transformers", "BERT",
                "GPT", "Computer Vision", "NLP", "Deep Learning", "Reinforcement Learning",
                "MLOps", "Model Deployment", "Feature Engineering", "Hugging Face"
            ],
            "Siber Güvenlik": [
                "Network Security", "Penetration Testing", "SIEM", "Firewall",
                "Cryptography", "Security Auditing", "Vulnerability Assessment",
                "Incident Response", "OWASP", "Kali Linux", "Wireshark", "Burp Suite"
            ],
            "Finans Teknolojileri": [
                "Python", "Java", "Blockchain", "Smart Contracts", "Solidity",
                "Risk Modeling", "Algorithmic Trading", "Financial Analysis",
                "Payment Gateway Integration", "Regulatory Compliance", "SQL"
            ],
            "E-Ticaret": [
                "SEO", "Google Analytics", "A/B Testing", "Product Management",
                "User Experience Design", "Conversion Optimization", "Shopify",
                "Magento", "Payment Integration", "Inventory Management", "CRM"
            ],
            "Sağlık Teknolojileri": [
                "HL7", "FHIR", "Medical Imaging", "Electronic Health Records",
                "HIPAA Compliance", "Bioinformatics", "Python", "R",
                "Clinical Data Management", "Telemedicine Platforms"
            ],
            "Oyun Geliştirme": [
                "Unity", "Unreal Engine", "C#", "C++", "3D Modeling", "Blender",
                "Game Physics", "Shader Programming", "Multiplayer Systems",
                "Game AI", "Animation", "Performance Optimization"
            ],
            "Bulut Bilişim": [
                "AWS", "Azure", "Google Cloud", "Terraform", "Docker",
                "Kubernetes", "Lambda", "S3", "EC2", "CloudFormation",
                "Serverless Architecture", "Cloud Security", "Cost Optimization"
            ],
            "IoT ve Gömülü Sistemler": [
                "C", "C++", "Python", "MQTT", "Arduino", "Raspberry Pi",
                "Embedded Linux", "RTOS", "Circuit Design", "Sensor Integration",
                "Wireless Communication", "PCB Design", "Microcontrollers"
            ]
        }
        
        # Deneyim seviyeleri
        self.experience_levels = ["Junior", "Mid-Level", "Senior", "Lead", "Principal"]
        
        # Şirket tipleri
        self.company_types = [
            "Startup", "Scale-up", "Kurumsal Firma", "Uluslararası Şirket",
            "Teknoloji Devi", "Danışmanlık Firması", "Ar-Ge Merkezi"
        ]
    
    def generate_job_postings(self, n: int = 5000) -> pd.DataFrame:
        """
        Sentetik iş ilanları oluşturur
        
        Args:
            n: Oluşturulacak ilan sayısı
            
        Returns:
            İş ilanlarını içeren DataFrame
        """
        job_postings = []
        
        for i in range(n):
            sector = random.choice(self.sectors)
            job_title = random.choice(self.job_titles[sector])
            experience_level = random.choice(self.experience_levels)
            company_type = random.choice(self.company_types)
            
            # Beceriler seçimi (5-12 arasında)
            sector_skills = self.skills[sector]
            num_skills = random.randint(5, min(12, len(sector_skills)))
            required_skills = random.sample(sector_skills, num_skills)
            
            # İş tanımı oluştur
            description = self._generate_job_description(
                job_title, sector, experience_level, required_skills, company_type
            )
            
            job_postings.append({
                "job_id": f"JOB_{i+1:05d}",
                "title": f"{experience_level} {job_title}",
                "sector": sector,
                "description": description,
                "required_skills": ", ".join(required_skills),
                "experience_level": experience_level,
                "company_type": company_type,
                "location": random.choice(["İstanbul", "Ankara", "İzmir", "Remote", "Hybrid"])
            })
        
        return pd.DataFrame(job_postings)
    
    def _generate_job_description(self, title: str, sector: str, 
                                   level: str, skills: List[str], 
                                   company_type: str) -> str:
        """İş tanımı metni oluşturur"""
        
        templates = [
            f"{company_type} bünyesinde görev yapacak {level} seviye {title} arıyoruz. "
            f"{sector} alanında deneyimli, {', '.join(skills[:3])} teknolojilerine hakim "
            f"ve takım çalışmasına yatkın adaylar aramaktayız. "
            f"Gerekli teknik beceriler: {', '.join(skills)}. "
            f"Projelerimizde aktif rol alacak, kod kalitesine önem verecek ve "
            f"sürekli öğrenmeye açık profesyoneller ile çalışmak istiyoruz.",
            
            f"{sector} sektöründe faaliyet gösteren {company_type}'ümüz için "
            f"{level} {title} pozisyonunda ekip arkadaşı arıyoruz. "
            f"Pozisyon gereksinimleri: {', '.join(skills[:5])} konularında deneyim. "
            f"Ayrıca {', '.join(skills[5:])} teknolojilerine aşinalık beklenmektedir. "
            f"Dinamik ve yenilikçi bir ortamda çalışmak isteyen adayları bekliyoruz.",
            
            f"{level} seviyede {title} pozisyonu için başvuruları kabul ediyoruz. "
            f"{company_type} olarak {sector} alanında öncü projeler geliştiriyoruz. "
            f"Aranan nitelikler: {', '.join(skills)}. "
            f"Problem çözme becerisi yüksek, analitik düşünebilen ve "
            f"teknolojiye tutkulu profesyoneller ile çalışmak istiyoruz."
        ]
        
        return random.choice(templates)
    
    def generate_sample_cvs(self, n: int = 10) -> pd.DataFrame:
        """
        Örnek CV metinleri oluşturur
        
        Args:
            n: Oluşturulacak CV sayısı
            
        Returns:
            CV'leri içeren DataFrame
        """
        cvs = []
        
        for i in range(n):
            primary_sector = random.choice(self.sectors)
            secondary_sectors = random.sample(
                [s for s in self.sectors if s != primary_sector], 
                k=min(2, len(self.sectors)-1)
            )
            
            # Beceriler (ana ve yan sektörlerden)
            primary_skills = random.sample(self.skills[primary_sector], 
                                          k=random.randint(6, 10))
            secondary_skills = []
            for sec_sector in secondary_sectors:
                secondary_skills.extend(
                    random.sample(self.skills[sec_sector], k=random.randint(2, 4))
                )
            
            all_skills = list(set(primary_skills + secondary_skills))
            
            # Deneyim yılı
            years_of_experience = random.randint(1, 15)
            
            # CV metni oluştur
            cv_text = self._generate_cv_text(
                primary_sector, all_skills, years_of_experience, secondary_sectors
            )
            
            cvs.append({
                "cv_id": f"CV_{i+1:03d}",
                "primary_sector": primary_sector,
                "cv_text": cv_text,
                "skills": ", ".join(all_skills),
                "years_of_experience": years_of_experience
            })
        
        return pd.DataFrame(cvs)
    
    def _generate_cv_text(self, primary_sector: str, skills: List[str],
                          years: int, secondary_sectors: List[str]) -> str:
        """CV metni oluşturur"""
        
        cv_template = f"""
        ÖZET: {years} yıllık deneyime sahip {primary_sector} profesyoneliyim. 
        {', '.join(skills[:5])} teknolojilerinde uzmanım ve 
        {', '.join(skills[5:8])} konularında ileri düzey bilgiye sahibim.
        
        TEKNİK BECERİLER:
        {', '.join(skills)}
        
        ÇALIŞMA DENEYİMİ:
        - {primary_sector} alanında {years} yıl profesyonel deneyim
        - {', '.join(secondary_sectors)} sektörlerinde proje deneyimi
        - Büyük ölçekli projelerde takım liderliği ve geliştirme deneyimi
        - Agile/Scrum metodolojileri ile çalışma tecrübesi
        
        EĞİTİM:
        - Bilgisayar Mühendisliği / Yazılım Mühendisliği Lisans
        - {random.choice(['Veri Bilimi', 'Yapay Zeka', 'Yazılım Geliştirme', 'Sistem Analizi'])} alanında sertifikalar
        
        PROJELER:
        - Çeşitli {primary_sector} projelerinde geliştirme ve mimari tasarım
        - {', '.join(skills[:3])} kullanarak ölçeklenebilir sistemler geliştirme
        - Performans optimizasyonu ve kod kalitesi iyileştirmeleri
        """
        
        return cv_template.strip()
    
    def save_data(self, job_postings: pd.DataFrame, cvs: pd.DataFrame, 
                  output_dir: str = "data"):
        """Oluşturulan verileri kaydeder"""
        job_postings.to_csv(f"{output_dir}/job_postings.csv", index=False, encoding='utf-8')
        cvs.to_csv(f"{output_dir}/sample_cvs.csv", index=False, encoding='utf-8')
        print(f"✓ {len(job_postings)} iş ilanı kaydedildi: {output_dir}/job_postings.csv")
        print(f"✓ {len(cvs)} CV kaydedildi: {output_dir}/sample_cvs.csv")


if __name__ == "__main__":
    # Test için
    generator = SyntheticDataGenerator()
    
    print("Sentetik veri oluşturuluyor...")
    jobs_df = generator.generate_job_postings(n=5000)
    cvs_df = generator.generate_sample_cvs(n=10)
    
    print(f"\n✓ {len(jobs_df)} iş ilanı oluşturuldu")
    print(f"✓ {len(cvs_df)} CV oluşturuldu")
    
    print("\nİlk 3 iş ilanı:")
    print(jobs_df[['job_id', 'title', 'sector']].head(3))
    
    print("\nİlk 2 CV:")
    print(cvs_df[['cv_id', 'primary_sector', 'years_of_experience']].head(2))