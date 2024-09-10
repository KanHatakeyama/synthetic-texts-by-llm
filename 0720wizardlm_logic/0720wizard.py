# %%
# %%
from llama_cpp import Llama
from datetime import datetime
import json
import os
import random
import time

for i in range(109):
    random.seed(datetime.now().time().microsecond+random.randint(0,10000))



# %%

wait_time = random.randint(1, 10)
time.sleep(wait_time)

#####################
# 設定関連
n_records = 1
out_dir = "0720out_data"

#tsubame
#model_path="/gs/fs/tga-hatakeyama/model/Mixtral-8x22B-Instruct-v0.1.Q8_0-00001-of-00004.gguf"
model_path="../model/WizardLM-2-8x22B.Q8_0-00001-of-00005.gguf"
#model_path="/gs/fs/tga-hatakeyama/model_wz/WizardLM-2-8x22B.Q8_0-00001-of-00005.gguf"

################
# メイン

os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/model_{current_time_no_symbols}.jsonl"

def is_abnormal_eng_text(text, threshold=40):
    words = text.split()
    word_count = len(words)
    period_count = text.count('.')
    ratio = word_count / period_count if period_count > 0 else word_count
    return ratio > threshold

class GGUFBot:
    def __init__(self, model_path="model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf",
                 max_new_tokens=4000,
                 n_gpu_layers=100,
                 n_ctx=4096) -> None:
        print("loading model...")

        self.model = Llama(model_path=model_path,
                           n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, )
        self.max_new_tokens = max_new_tokens

    def ask(self, question):

        prompt = f"""<s>[INST]{question}[/INST]"""

        output = self.model(
            prompt,
            max_tokens=self.max_new_tokens,
            # temperature = 0.7,
            # top_p = 0.8,
            # repeat_penalty = 1.1,
            # frequency_penalty = 1.0,
            # presence_penalty = 1.0,
            # stop = ["\n###  Instruction:", "\n### Response:", "\n"],
            # echo = True,
        )
        return output["choices"][0]["text"].strip()


bot = GGUFBot(model_path=model_path)


# %%


print("initiated llm")

genre_texts="""Literature

Fiction
Contemporary Fiction
Literary Fiction
Historical Fiction
Mystery Fiction
Crime Fiction
Gothic Fiction
Magical Realism
Dystopian Fiction
Utopian Fiction
Noir Fiction
Political Fiction
Speculative Fiction
Urban Fiction
Picaresque Fiction
Western Fiction
Non-fiction
True Crime
Self-Help
Health and Wellness
Business and Economics
Politics and Government
Science and Nature
History
Travel
Essays
Criticism
Journalism
Philosophy
Religion and Spirituality
Parenting
Education

Music

Classical
Baroque
Romantic
Modern Classical
Contemporary Classical
Chamber Music
Symphonic Music
Opera
Choral Music
Jazz
Bebop
Swing
Free Jazz
Smooth Jazz
Fusion
Latin Jazz
Vocal Jazz
Jazz Funk
Avant-Garde Jazz
Blues
Delta Blues
Chicago Blues
Electric Blues
Country Blues
Blues Rock
Texas Blues
Soul Blues
Piedmont Blues
Rock
Classic Rock
Alternative Rock
Punk Rock
Progressive Rock
Hard Rock
Indie Rock
Garage Rock
Glam Rock
Grunge
Psychedelic Rock
Southern Rock
Pop
Teen Pop
Pop Rock
Bubblegum Pop
Dance Pop
Synthpop
Indie Pop
Electropop
Art Pop
Hip-Hop
Rap
Trap
Boom Bap
Conscious Hip-Hop
Gangsta Rap
Alternative Hip-Hop
Crunk
Lofi Hip-Hop
Jazz Rap

Film & TV

Action
Superhero
Martial Arts
War
Disaster
Spy
Crime
Western
Adventure
Survival
Comedy
Romantic Comedy
Satire
Slapstick
Parody
Dark Comedy
Screwball Comedy
Stand-Up Comedy
Teen Comedy
Musical Comedy
Drama
Legal Drama
Medical Drama
Period Drama
Family Drama
Teen Drama
Political Drama
Psychological Drama
Social Drama
Crime Drama
Horror
Gothic Horror
Paranormal
Slasher
Monster
Psychological Horror
Body Horror
Found Footage
Zombie
Folk Horror
Sci-Fi
Space Opera
Cyberpunk
Time Travel
Alien Invasion
Dystopian
Utopian
Steampunk
Post-Apocalyptic
Hard Science Fiction
Fantasy
High Fantasy
Urban Fantasy
Dark Fantasy
Epic Fantasy
Sword and Sorcery
Contemporary Fantasy
Historical Fantasy
Fairy Tale
Mythic Fantasy

Art

Abstract
Abstract Expressionism
Color Field
Geometric Abstraction
Lyrical Abstraction
Abstract Landscape
Minimalist Abstraction
Hard-Edge Painting
Non-Objective Art
Action Painting
Realism
Photorealism
Social Realism
Hyperrealism
Classical Realism
Contemporary Realism
Naturalism
Magical Realism
Impressionistic Realism
Urban Realism
Surrealism
Automatism
Biomorphic
Dream Imagery
Hypnagogic Art
Metaphysical Art
Fantastic Art
Symbolism
Abstract Surrealism
Figurative Surrealism
Cubism
Analytical Cubism
Synthetic Cubism
Crystal Cubism
Orphic Cubism
Curvilinear Cubism
Proto-Cubism
Prismatic Cubism
Magic Cubism
Late Cubism
Impressionism
Post-Impressionism
Neo-Impressionism
Modern Impressionism
Contemporary Impressionism
American Impressionism
French Impressionism
Pointillism
Divisionism
Symbolist Impressionism

Games

Action
Platformers
Shooters
Beat 'em Ups
Rhythm Games
Fighting Games
Stealth Games
Survival Games
Hack and Slash
Battle Royale
Adventure
Text Adventures
Graphic Adventures
Visual Novels
Interactive Movies
Puzzle Adventures
Survival Horror
Action-Adventure
Narrative Adventures
Metroidvania
Role-playing (RPG)
Action RPG
Tactical RPG
Japanese RPG (JRPG)
Western RPG (WRPG)
Sandbox RPG
Dungeon Crawler
MMORPG
Roguelike
Open World RPG
Simulation
Life Simulation
Vehicle Simulation
Construction and Management Simulation
Farming Simulation
Dating Simulation
Sports Simulation
Flight Simulation
Business Simulation
Medical Simulation
Strategy
Real-Time Strategy (RTS)
Turn-Based Strategy (TBS)
Tower Defense
Grand Strategy
4X Strategy
Tactical Strategy
MOBA
City-Building Games
Card Battler
Sports
Racing Games
Football Games
Basketball Games
Baseball Games
Golf Games
Tennis Games
Skateboarding Games
Boxing Games
Fishing Games
Puzzle
Match-3 Games
Logic Puzzles
Hidden Object Games
Physics Puzzles
Tile-Matching Games
Trivia Games
Sudoku
Crossword Puzzles
Jigsaw Puzzles

Science

Physics
Theoretical Physics
Experimental Physics
Astrophysics
Quantum Physics
Particle Physics
Condensed Matter Physics
Nuclear Physics
Plasma Physics
Biophysics
Chemistry
Organic Chemistry
Inorganic Chemistry
Physical Chemistry
Analytical Chemistry
Biochemistry
Theoretical Chemistry
Polymer Chemistry
Environmental Chemistry
Medicinal Chemistry
Biology
Cell Biology
Molecular Biology
Evolutionary Biology
Marine Biology
Microbiology
Genetics
Zoology
Botany
Ecology
Astronomy
Observational Astronomy
Theoretical Astronomy
Planetary Science
Stellar Astronomy
Galactic Astronomy
Cosmology
Astrobiology
Radio Astronomy
Infrared Astronomy
Geology
Petrology
Mineralogy
Volcanology
Seismology
Paleontology
Geophysics
Hydrogeology
Environmental Geology
Geochemistry
Environmental Science
Conservation Biology
Environmental Chemistry
Environmental Physics
Ecology
Atmospheric Science
Oceanography
Soil Science
Environmental Engineering
Climate Science
Medicine
Internal Medicine
Surgery
Pediatrics
Psychiatry
Neurology
Dermatology
Cardiology
Oncology
Pathology

Technology

Artificial Intelligence
Machine Learning
Deep Learning
Natural Language Processing
Computer Vision
Reinforcement Learning
Robotics
Expert Systems
Fuzzy Logic
Evolutionary Computation
Software Development
Web Development
Mobile Development
Game Development
Desktop Applications
Embedded Systems
Cloud Computing
DevOps
Agile Methodologies
Software Testing
Cybersecurity
Network Security
Application Security
Information Security
Cyber Threat Intelligence
Cryptography
Ethical Hacking
Incident Response
Security Operations
Digital Forensics
Networking
Network Architecture
Network Protocols
Wireless Networks
Network Security
Cloud Networking
Software-Defined Networking (SDN)
Network Automation
Internet of Things (IoT)
Network Monitoring
Data Science
Data Analysis
Data Engineering
Data Mining
Big Data
Literature

Fiction
Non-fiction
Fantasy
Science Fiction
Mystery
Thriller
Romance
Historical Fiction
Horror
Biography
Autobiography
Memoir
Poetry
Drama
Satire
Adventure

Music

Classical
Jazz
Blues
Rock
Pop
Hip-Hop
Country
Reggae
Electronic
Folk
R&B
Metal
Punk
Soul
Gospel
Opera

Film & TV

Action
Comedy
Drama
Horror
Sci-Fi
Fantasy
Romance
Thriller
Mystery
Documentary
Animation
Adventure
Musical
Crime
Western
Historical

Art

Abstract
Realism
Impressionism
Surrealism
Cubism
Expressionism
Pop Art
Minimalism
Conceptual Art
Street Art
Digital Art
Photorealism
Dadaism
Baroque
Rococo

Games

Action
Adventure
Role-playing (RPG)
Simulation
Strategy
Sports
Puzzle
Horror
Educational
Sandbox
MMO (Massively Multiplayer Online)
Racing
Fighting
Shooter
Platformer

Sports

Soccer
Basketball
Baseball
Tennis
Golf
American Football
Cricket
Rugby
Boxing
Wrestling
Martial Arts
Swimming
Track and Field
Cycling
Gymnastics
Skiing

Technology

Artificial Intelligence
Robotics
Software Development
Cybersecurity
Networking
Blockchain
Virtual Reality
Augmented Reality
Internet of Things (IoT)
Cloud Computing
Data Science
Machine Learning
Quantum Computing
Renewable Energy
Biotechnology

Food & Drink

Italian
Chinese
Japanese
French
Mexican
Indian
Thai
Mediterranean
American
Vegan
Vegetarian
Seafood
Barbecue
Bakery
Fast Food

Fashion

Haute Couture
Ready-to-Wear
Streetwear
Sportswear
Casual
Formal
Vintage
Avant-garde
Sustainable Fashion
Ethnic Fashion
Business Attire
Lingerie
Footwear
Accessories
Outerwear

Science

Physics
Chemistry
Biology
Astronomy
Geology
Environmental Science
Medicine
Genetics
Neuroscience
Ecology
Meteorology
Oceanography
Paleontology
Anthropology
Psychology

Business & Economics

Finance
Marketing
Management
Entrepreneurship
Economics
Real Estate
International Business
Human Resources
Operations Management
Supply Chain
E-commerce
Corporate Strategy
Business Ethics
Investment
Accounting

Education

Early Childhood Education
Primary Education
Secondary Education
Higher Education
Adult Education
Special Education
Educational Technology
Curriculum Development
Educational Psychology
Educational Administration
Vocational Training
Online Learning
Language Learning
STEM Education
Arts Education

"""

genre_list=genre_texts.split("\n")
genre_list=[i for i in genre_list if i!=""]

def prepare_records(
                    n_records=300,
                    ):
    records = []

    for i in range(n_records):
        level = random.choice(["junior high-school", "senior high-school", "university",
                               "graduate school", "professional","elementary school"
                               ])
        genre=random.choice(genre_list)
        prompt=f"""Output a {level} level, random logical quiz and answer.
- Never include numbers in the question.
- Genre: {genre}
- The question should be complex and require a step-by-step solution.
- The question must be a logical question."""
        records.append(
            {"prompt": prompt,}
        )

    return records


while True:

    # 回答
    records = prepare_records(
       n_records=n_records,
    )
    prompts = [record["prompt"] for record in records]

    outputs1 = []
    for prompt in prompts:
        outputs1.append(bot.ask(prompt))

    for record, output in zip(records, outputs1):
        if is_abnormal_eng_text(output):
            continue
        if output == "":
            continue

        output=output.replace("[TEXTUAL SOLUTION PROCESS]:","")
        output=output.replace("[Textual Solution Process]:","")
        output=output.strip()

        record["text"]=output
        record.pop("prompt")

        # print("saving to "+out_path)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
