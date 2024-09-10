# %%
# %%
# %%
import os
from datetime import datetime
from vllm import SamplingParams, LLM
import json
import random
import re
import sys
args = sys.argv
import time


# job idを取得
# job_id=os.environ['$SLURM_JOB_ID']
#job_id = args[1]


def get_longest_phrase_length(text):
    # 区切り文字として、スペース、カンマ、句読点、改行を指定
    delimiters = r'[ ,。！？、\n]'
    # テキストを区切り文字で分割
    try:
        phrases = re.split(delimiters, text)
        # 最大のフレーズの長さを取得
        max_length = max(len(phrase) for phrase in phrases)
    except:
        max_length=9999
    return max_length

def is_abnormal_text(text, threshold=40):
    words = text.split()
    word_count = len(words)
    # 複数の区切り文字をカウント
    period_count = text.count('.') + text.count(',') + text.count('､') + text.count('｡')
    ratio = word_count / period_count if period_count > 0 else word_count
    return ratio > threshold

batch_size=200
max_count=10**5
out_dir = "0725wizard_7b"

pid = os.getpid()
seed=int(pid)+int(datetime.now().timestamp())
print("seed: ",seed)
random.seed(seed)


# %%



# %%

# %%
os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
rand_id=random.randint(0,10000)


# %%

model_name = "dreamgen/WizardLM-2-7B"
tensor_parallel_size=1
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=4000,
          # max_model_len=7000,
         #  gpu_memory_utilization=0.9,
         tensor_parallel_size=tensor_parallel_size,
          )

def llm_gen(llm,prompt_list,temperature=0.7,top_k=50):

    outputs = llm.generate(
        prompt_list,
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=2048,
            repetition_penalty=1.2,
            top_k=top_k,
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]


# %%
def question_to_prompt(question,role="an artificial intelligence assistant"):
    prompt=f"""A chat between a curious user and {role}. The assistant gives helpful, 
detailed, and polite answers to the user's questions. USER: {question} ASSISTANT:"""
    return prompt

# %%
jobs="""Company employee
Public servant
Self-employed
Animator
Accountant
Actor
Actuary
Acupuncturist
Administrative Assistant
Advertising Manager
Aerospace Engineer
Agricultural Engineer
Agricultural Inspector
Agricultural Manager
Air Traffic Controller
Aircraft Mechanic
Airline Pilot
Animator
Anthropologist
App Developer
Appraiser
Architect
Archivist
Art Director
Art Historian
Art Restorer
Art Therapist
Artist
Astronomer
Athletic Trainer
Auctioneer
Auditor
Auto Mechanic
Automotive Designer
Baker
Bank Teller
Barber
Bartender
Biochemist
Biologist
Biomedical Engineer
Blogger
Book Editor
Bookkeeper
Botanist
Brewer
Broadcast Technician
Budget Analyst
Building Inspector
Business Analyst
Business Consultant
Business Development Manager
Butcher
Call Center Representative
Carpenter
Cartographer
Casino Dealer
Chef
Chemical Engineer
Chemist
Childcare Worker
Chiropractor
Civil Engineer
Claims Adjuster
Clinical Psychologist
Coach
Commercial Diver
Commercial Pilot
Community Organizer
Computer Programmer
Computer Scientist
Computer Support Specialist
Construction Manager
Construction Worker
Consultant
Content Creator
Copywriter
Corporate Trainer
Corrections Officer
Cosmetologist
Counselor
Court Reporter
Creative Director
Crime Scene Investigator
Criminal Lawyer
Curator
Customer Service Representative
Dance Instructor
Data Analyst
Data Scientist
Database Administrator
Dentist
Dermatologist
Detective
Dietitian
Director of Photography
Disc Jockey
Dog Trainer
Drafter
Economist
Editor
Education Administrator
Electrician
Elementary School Teacher
Emergency Medical Technician (EMT)
Energy Consultant
Engineer
Environmental Consultant
Environmental Engineer
Environmental Scientist
Epidemiologist
Event Planner
Executive Assistant
Family Therapist
Fashion Designer
Fashion Model
Film Director
Financial Advisor
Financial Analyst
Financial Planner
Firefighter
Fishery Worker
Florist
Food Scientist
Forensic Scientist
Forest Ranger
Funeral Director
Furniture Designer
Game Designer
Game Developer
Genetic Counselor
Geneticist
Geographer
Geologist
Graphic Designer
Guidance Counselor
Hairstylist
Health and Safety Officer
Health Educator
Healthcare Administrator
Historian
Home Inspector
Home Stager
Horticulturist
Hospital Administrator
Hotel Manager
Human Resources Manager
Illustrator
Industrial Designer
Industrial Engineer
Information Security Analyst
Information Technology (IT) Specialist
Insurance Agent
Interior Designer
Interpreter
Investment Banker
Journalist
Judge
Kindergarten Teacher
Laboratory Technician
Land Surveyor
Landscape Architect
Law Clerk
Lawyer
Librarian
Licensed Practical Nurse (LPN)
Life Coach
Locksmith
Logistician
Machine Operator
Magazine Editor
Makeup Artist
Management Consultant
Manufacturing Engineer
Marine Biologist
Market Research Analyst
Marketing Manager
Massage Therapist
Mathematician
Mechanical Engineer
Media Buyer
Medical Assistant
Medical Billing Specialist
Medical Coder
Medical Laboratory Technician
Medical Records Technician
Medical Transcriptionist
Mental Health Counselor
Meteorologist
Microbiologist
Midwife
Mining Engineer
Mobile App Developer
Model
Mortgage Broker
Motor Vehicle Inspector
Music Producer
Music Teacher
Musician
Nanny
Network Administrator
Network Engineer
Neurologist
News Anchor
Nurse
Nurse Practitioner
Nutritionist
Occupational Therapist
Oceanographer
Office Manager
Oncologist
Operations Manager
Optician
Optometrist
Orthodontist
Orthopedic Surgeon
Painter
Paralegal
Park Ranger
Patent Examiner
Pathologist
Pediatrician
Personal Trainer
Pharmacist
Photographer
Physical Therapist
Physician
Physician Assistant
Physiotherapist
Pilot
Plumber
Police Officer
Political Scientist
Politician
Postal Worker
Preschool Teacher
Private Investigator
Probation Officer
Product Manager
Professional Athlete
Professional Speaker
Programmer
Project Manager
Property Manager
Psychiatrist
Psychologist
Public Relations Specialist
Public Speaker
Quality Assurance Analyst
Radiologic Technologist
Radiologist
Real Estate Agent
Real Estate Appraiser
Real Estate Broker
Receptionist
Registered Nurse (RN)
Reporter
Research Scientist
Respiratory Therapist
Restaurant Manager
Retail Buyer
Retail Salesperson
Risk Manager
Robotics Engineer
Sales Manager
Sales Representative
School Counselor
School Principal
Screenwriter
Sculptor
Security Guard
Security Specialist
Social Media Manager
Social Worker
Software Developer
Software Engineer
Software Tester
Solar Panel Installer
Sound Engineer
Speech Pathologist
Statistician
Stockbroker
Store Manager
Structural Engineer
Surgeon
Surveyor
Systems Analyst
Tax Advisor
Taxidermist
Teacher
Technical Writer
Telecommunications Specialist
Television Producer
Travel Agent
Translator
Transportation Planner
Travel Writer
Truck Driver
Ultrasound Technician
Underwriter
Urban Planner
UX Designer
Veterinarian
Video Editor
Videographer
Virologist
Voice Actor
Waiter/Waitress
Warehouse Manager
Web Designer
Web Developer
Wedding Planner
Wildlife Biologist
Writer
Yoga Instructor
Zoologist
Air Conditioning Technician
Archaeologist
Art Critic
Artificial Intelligence Engineer
Astronaut
Auctioneer
Auto Detailer
Aviation Inspector
Biostatistician
Blacksmith
Blogger
Brewmaster
Call Center Operator
Career Counselor
Chiropractor
Climatologist
College Professor
Computer Forensics Analyst
Corporate Lawyer
Court Clerk
Cruise Director
Cryptographer
Cybersecurity Analyst
Dance Choreographer
Database Architect
Demographer
Dental Hygienist
Dialysis Technician
Dietetic Technician
Digital Marketing Specialist
Diplomat
Document Controller
Editorial Assistant
Educational Consultant
Electric Line Worker
Energy Auditor
Environmental Health Specialist
Estate Planner
Event Coordinator
Executive Chef
Family Lawyer
Fashion Stylist
Film Editor
Fine Artist
Fish and Game Warden
Fitness Instructor
Floral Designer
Forensic Accountant
Fundraising Manager
Genealogist
Geoscientist
Glass Blower
Graphic Novelist
Green Building Architect
Hair Colorist
Health Coach
Heavy Equipment Operator
Hotel Concierge
HVAC Technician
Hydrologist
Immigration Lawyer
Import/Export Specialist
Industrial Hygienist
Interior Decorator
International Relations Specialist
Interpreter for the Deaf
IT Project Manager
Jewelry Designer
Kindergarten Teacher
Landscape Designer
Laser Technician
Legal Secretary
Lighting Designer
Litigation Support Specialist
Loan Officer
Logistics
Ecologist
Appraiser
Bartender
Construction Technologist
Dancer
Environmental Engineer
Tutor
Game Designer
Real Estate Agent
Translator
Gardener
Chef
Financial Analyst
HR Specialist
Landscape Architect
Nutritionist
Occupational Therapist
Park Ranger
Personal Trainer
Pilot
Public Relations Specialist
Radiologist
Speech Therapist
Urban Planner
Web Developer
Wildlife Biologist
Yoga Instructor
Zoologist
Doctor
Nurse
Engineer
Designer
Teacher
Salesperson
Actor
Agricultural worker
Artist
Athlete
Company employee
Consultant
Construction worker
Entertainer
Finance industry worker
Fishery worker
Homemaker
Insurance industry worker
IT industry worker
Manufacturing worker
Musician
Real estate industry worker
Researcher
Service industry worker
Student
Transport worker
Unemployed
Writer
Accountant
Architect
Chef/Cook
Dentist
Electrician
Event Planner
Farmer
Firefighter
Graphic Designer
Journalist
Lawyer
Librarian
Marketing Specialist
Mechanic
Pharmacist
Photographer
Physical Therapist
Police Officer
Plumber
Psychologist
Scientist
Social Worker
Software Developer
Veterinarian
Self-employed
Doctor
Nurse
Engineer
Designer
Teacher
Salesperson
Actor
Agricultural worker
Artist
Athlete
Company employee
Consultant
Construction worker
Designer
Doctor
Engineer
Entertainer
Finance industry worker
Fishery worker
Homemaker
Insurance industry worker
IT industry worker
Manufacturing worker
Musician
Nurse
Public servant
Real estate industry worker
Researcher
Salesperson
Self-employed
Service industry worker
Student
Teacher
Transport worker
Unemployed
Writer
Service industry worker
Agricultural worker
Fishery worker
Construction worker
Manufacturing worker
Transport worker
Finance industry worker
Insurance industry worker
Real estate industry worker
IT industry worker
Consultant
Writer
Artist
Musician
Actor
Accountant
Architect
Chef/Cook
Dentist
Electrician
Event Planner
Farmer
Firefighter
Graphic Designer
Journalist
Lawyer
Librarian
Marketing Specialist
Mechanic
Pharmacist
Photographer
Physical Therapist
Police Officer
Plumber
Psychologist
Scientist
Social Worker
Software Developer
Veterinarian
Entertainer
Athlete
Researcher
Student
Homemaker
Unemployed
"""
job_list=jobs.split("\n")
job_list=[i for i in job_list if i!=""]

character_text="""
Emotionally intelligent
Empathetic: Able to understand and share the feelings of others.
Visionary: Having a clear and inspiring vision for the future.
Strategic: Skilled in planning and executing complex plans.
Humorous: Possesses a good sense of humor and can lighten the mood.
Pragmatic: Focused on practical and realistic solutions.
Inquisitive: Naturally curious and eager to explore new ideas.
Supportive: Provides encouragement and assistance to others.
Decisive: Capable of making decisions quickly and effectively.
Tolerant: Respects and accepts different opinions and behaviors.
Resilient: Able to recover quickly from setbacks.
Meticulous: Pays attention to even the smallest details.
Confident: Self-assured and able to assert oneself.
Diplomatic: Skilled in managing negotiations and sensitive situations.
Intuitive: Able to understand things instinctively.
Balanced: Maintains a good balance between different aspects of life.
Innovative: Continuously coming up with new and creative ideas.
Assertive: Confidently expresses opinions and needs.
Charismatic: Attracts and inspires others with their personality.
Flexible: Easily adapts to new situations and changes.
Patient: Remains calm and composed, even in difficult situations.
Persevering: Continues to work towards goals despite obstacles.
Observant: Notices and understands details that others may miss.
Motivational: Inspires and motivates others to achieve their best.
Empowered: Feels confident in taking control and making decisions.
Generous: Willing to share time, resources, or knowledge with others.
Spontaneous: Open to unexpected changes and ready to take advantage of them.
Organized: Keeps things well-structured and orderly.
Insightful: Provides deep and accurate understanding of complex issues.
Persuasive: Able to convince others to see things from their perspective.
Self-motivated: Driven by internal goals and desires to succeed.
Optimistic: Maintains a positive outlook on life and situations.
Punctual: Values time and ensures timely completion of tasks.
Self-disciplined: Able to control oneself and stick to plans or goals.
Goal-oriented: Focused on achieving specific objectives.
Innovative thinker: Thinks outside the box and develops new solutions.
Reflective: Takes time to think deeply about experiences and decisions.
Approachable: Easy to talk to and open to communication with others.
Cultivates good relationships: Builds and maintains strong, positive relationships.
Values diversity: Appreciates and embraces differences in people and perspectives.
Responsible: Takes ownership of actions and their consequences.
Self-aware: Has a good understanding of one's strengths and weaknesses.
Generates positive energy: Brings positivity and enthusiasm to the environment.
Collaborative: Works well with others to achieve common goals.
Detail-oriented: Focuses on and cares about the finer aspects of a task.
Innovative mindset: Constantly looks for new and better ways of doing things.
Strong sense of responsibility
Creative
Compassionate
Highly curious
Excellent communication skills
Capable of logical thinking
Highly adaptable
Diligent
Strong ethical sense
Cooperative
Attentive to detail
High problem-solving ability
Optimistic
Passionate
Moderately confident
Honest
Objective
Open-minded
Sense of humor
Persistent
Culturally sensitive
Patient
Proactive
Innovative thinker
Leadership qualities
Resilient
Respectful of diversity
Team player
Humble
Disciplined
Quick to respond
Conscientious decision-maker
Considerate
Respectful towards others
Positive outlook
Self-controlled
Trustworthy
Resourceful
Analytical
Eager to learn
Kind
Dedicated
Compassionate
Enthusiastic
Work enthusiast
Rich in creativity
Keen
Admirable
High ethical standards
Generate innovative ideas
Capable of strategic thinking
Thoughtful
Insightful
Modest
Passionate
Positive attitude
Attentive to small details
Reliable
Sociable
Independent
Always seeking improvement
Proactive
Strong-willed
Decisive
Compassionate
Actively problem-solving
Suggest creative ideas
Eager to acquire knowledge
Approachable
Communicative
Proactively tackle challenges
Eager to try new things
Demonstrate strong leadership
Always willing to learn
Contribute to the team
Pay attention to small details
Express opinions actively
Actively resolve issues
Eager to acquire new skills
Demonstrate teamwork
Actively tackle problems
Propose new ideas
Take on new challenges
Strong sense of responsibility
Pay attention to small details
Eager to learn new technologies
Adapt to new environments
Approach tasks with strong willpower
Eager for new experiences
Rough
Bad-tempered
Nitpicky
Difficult to deal with
"""
character_list=character_text.split("\n")
character_list=[i for i in character_list if i!=""]

# %%
genre_text="""Absolute value equations and inequalities
addition
Addition and subtraction in word problems
Advanced graph and data analysis (histograms, line graphs, etc.)
Advanced problems on area and volume (cylinders, triangular areas)
Advanced Topics
Algorithm problems
Applications of equations in optimization problems
Applications of multiplication and division in real-life scenarios
Applications of quadratic equations in word problems
Applications of ratios and proportions
Area and circumference of circles
Area and perimeter of shapes
Area of triangles and quadrilaterals
Basic addition and subtraction
Basic addition and subtraction with decimals and fractions
Basic concepts of fractions
Basic concepts of probability problems
Basic concepts Python problems
Basic graph reading
Basic statistics (mean, median, range)
Calculations of area and volume (squares, rectangles, cubes, cuboids)
Calculations with fractions and decimals
Comparison of lengths and weights
Composite calculations with addition and subtraction of integers
Composite calculations with fractions, decimals, and integers
Concepts of straight lines, parallel lines, and perpendicular lines
Conditional probability problems
Creating and interpreting tables and graphs
Creating and using APIs (RESTful)
Creating graphical user interfaces (GUI) (Tkinter)
Creating graphs (Matplotlib)
Creating interactive dashboards (Dash)
Creating web applications (Flask)
Differential equations basics
Division
Division with remainders
division.
Equations
Equations involving square roots and other radicals
Equations with logarithms and exponents
Estimation in addition and subtraction
Four arithmetic operations with decimals and fractions
High school-level logic problems
High school-level mathematics problems
High school-level thinking quizzes
Introduction to mental math strategies for addition and subtraction
Introduction to multiplication (memorizing multiplication tables)
Large numbers (up to 1 million)
Long multiplication and division
Measurement and comparison of angles
Middle school-level geometry problems
Middle school-level logic problems
Middle school-level mathematics problems
Middle school-level thinking quizzes
Multiples and factors
Multiplication
Multiplication and Division
Multiplication and division with decimals and fractions
Multiplication and division word problems
Nonlinear equations and iterative methods for solving them
Numbers up to 1,000
Numbers up to 10,000
Other Python problems for algorithms
Probability problems
Problems involving complex logic
Problems involving inequalities and their solutions
Problems involving parametric equations
Problems on addition and subtraction of polynomials
Problems on application problems with equations
Problems on applications of quadratic equations
Problems on applications of quadratic functions
Problems on basic addition and subtraction
Problems on basic geometric shapes (points, lines, parallel lines, angles, etc.)
Problems on basic multiplication and division
Problems on calculating and using expressions
Problems on calculations of expressions
Problems on characters and expressions
Problems on cross-sections and projections of three-dimensional figures
Problems on equations
Problems on equations in two variables
Problems on equations involving rational expressions
Problems on equations with complex numbers
Problems on equations with decimals
Problems on equations with fractions
Problems on exponential equations
Problems on four arithmetic operations (addition, subtraction, multiplication, division)
Problems on graphs and properties of inverse proportionality
Problems on graphs and properties of proportionality
Problems on graphs and properties of quadratic functions
Problems on independent and mutually exclusive events
Problems on irrational and rational numbers
Problems on linear functions
Problems on linear programming
Problems on monomials and polynomials
Problems on multiplication and division of monomials
Problems on nets and surface areas
Problems on parallel lines and geometric figures
Problems on partial fraction decomposition
Problems on plane figures
Problems on polynomial equations
Problems on positive and negative numbers
Problems on properties of geometric shapes
Problems on properties of parallel lines and angles
Problems on proportionality and inverse proportionality
Problems on quadratic equations
Problems on quadratic functions
Problems on simple linear equations
Problems on solving equations graphically
Problems on solving equations using matrices and determinants
Problems on solving linear equations
Problems on solving quadratic equations (factoring, completing the square, quadratic formula)
Problems on solving systems of linear equations
Problems on systems of linear equations
Problems on the application of linear functions
Problems on the basics of coordinates and vectors
Problems on the concept of positive and negative numbers
Problems on the concept of similarity
Problems on the concept of square roots
Problems on the congruence conditions of triangles
Problems on the graph of linear functions and their properties
Problems on the interior and exterior angles of polygons
Problems on the method of undetermined coefficients
Problems on the properties and applications of equations in coordinate geometry
Problems on the properties of circles and tangents
Problems on the properties of triangles and quadrilaterals
Problems on the surface area and volume of cylinders, cones, and spheres
Problems on three-dimensional figures (cubes, cuboids, etc.)
Problems on three-dimensional shapes
Problems on triangles and quadrilaterals
Problems on types and properties of quadrilaterals
Problems on variables and constants
Problems testing logical thinking
Problems using graphs to solve
Programming problems
Progress in addition and subtraction (carrying and borrowing)
Progress in multiplication and division
Properties of parallelograms and trapezoids
Python problems for adding code comments
Python problems for approximation algorithms
Python problems for arrays
Python problems for automating regular tasks (cron jobs, scheduled tasks)
Python problems for automation and scripting
Python problems for backtracking
Python problems for basic concepts of parallel processing
Python problems for Big O notation
Python problems for binary search
Python problems for breadth-first search (BFS)
Python problems for bubble sort
Python problems for complexity and efficiency
Python problems for creating API documentation
Python problems for creating batch processing scripts
Python problems for creating command-line interfaces (CLI)
Python problems for creating README files
Python problems for creating user interfaces
Python problems for data aggregation and analysis
Python problems for data cleaning and preprocessing
Python problems for data preprocessing and feature engineering
Python problems for data processing and transformation
Python problems for data structures
Python problems for data transformation (type conversion, format conversion)
Python problems for data visualization
Python problems for debugging and testing
Python problems for debugging code
Python problems for depth-first search (DFS)
Python problems for distributed computing
Python problems for divide-and-conquer algorithms
Python problems for documentation
Python problems for dynamic programming
Python problems for graphs (adjacency matrix, adjacency list)
Python problems for greedy algorithms
Python problems for heap sort
Python problems for heaps
Python problems for Huffman coding
Python problems for implementing algorithms
Python problems for implementing authentication and authorization
Python problems for implementing encryption and data security
Python problems for implementing graph algorithms (shortest path problems, minimum spanning trees, etc.)
Python problems for implementing search algorithms (binary search, depth-first search, etc.)
Python problems for implementing sorting algorithms (bubble sort, quicksort, etc.)
Python problems for insertion sort
Python problems for interpreting and visualizing results
Python problems for linear search
Python problems for lists
Python problems for machine learning and data science
Python problems for matrix operations
Python problems for memory optimization
Python problems for merge sort
Python problems for metaheuristics (genetic algorithms, simulated annealing)
Python problems for minimum spanning trees (Kruskal's algorithm, Prim's algorithm)
Python problems for multithreaded algorithms
Python problems for networks and security
Python problems for numerical algorithms
Python problems for optimizing computation speed
Python problems for parallel algorithms
Python problems for pattern matching (KMP algorithm, Rabin-Karp algorithm)
Python problems for performance optimization
Python problems for prime number testing
Python problems for profiling code
Python problems for queues
Python problems for quicksort
Python problems for randomized algorithms
Python problems for reading and writing CSV files
Python problems for reading and writing data
Python problems for reading and writing data from databases
Python problems for reading and writing JSON files
Python problems for reading and writing text files
Python problems for search algorithms
Python problems for selection sort
Python problems for setting up and running continuous integration (CI)
Python problems for shortest path problems (Bellman-Ford algorithm)
Python problems for socket programming
Python problems for sorting algorithms
Python problems for stacks
Python problems for string processing
Python problems for suffix trees
Python problems for the basic concepts of greedy algorithms
Python problems for the Euclidean algorithm
Python problems for the knapsack problem
Python problems for the longest common subsequence (LCS)
Python problems for trees (binary trees, binary search trees, AVL trees, etc.)
Python problems for trie trees
Python problems for version control
Python problems for web development
Python problems for web scraping
Python problems on algorithm design paradigms
Python problems on what algorithms are
Range of numbers and integer calculations
Recognizing numbers (0 to 100)
Recognizing shapes (circles, triangles, quadrilaterals, etc.)
Relationship between time, distance, and speed
Simple data organization and representation
Simple probability calculation problems
Simple problem-solving
Simple problem-solving using addition and subtraction
Solving linear equations with one variable
Solving quadratic equations (factoring, completing the square, quadratic formula)
Solving systems of equations using substitution and elimination methods
Square root problems
subtraction
Systems of inequalities
Systems of linear equations
Time calculations (minutes, seconds)
Time calculations and unit conversions
Training and evaluating machine learning models (Scikit-learn)
Understanding and measuring angles
Understanding and solving equations using numerical methods
Understanding and solving equations with variables on both sides
Understanding and solving problems involving factors and multiples
Understanding and visualizing parallel, symmetric, and rotational figures
Understanding fractions (1/2, 1/3, etc.)
Understanding integers and decimals
Understanding ratios and proportions
Understanding time (reading clocks)
Unit conversions and their applications
Using remote repositories (GitHub)
Writing and executing unit tests (unittest)

"""
genre_list=genre_text.split("\n")
genre_list=[i for i in genre_list if i!=""]

# %%
levels=[
"elementary school",
"junior high school",
"high school",
]

count=0
file_id=0
while True:
    seed=int(pid)+int(datetime.now().timestamp())
    print("seed: ",seed)
    random.seed(seed)
    parallel_conversations=[{"qid":i,"conversations":[]} for i in range(batch_size)]
    prompt_list=[]


    for qid in range(len(parallel_conversations)):
        job=random.choice(job_list)
        character=random.choice(character_list)
        role=f"You are {job} and {character}."
        genre=random.choice(genre_list)+","+random.choice(genre_list)
        level=random.choice(levels)
        command=f"""Prepare a mathematical, reasoning, logical, or coding problem or quiz, and its solution.
- Topic: {genre}.
- Level: {level}.
"""
        prompt_list.append(question_to_prompt(command,role))

    print(prompt_list[:3])  
    sentence_list=llm_gen(llm,prompt_list,temperature=0.01)

    #書き出し    
    for sentence in sentence_list:
        text=""
        remove_flag=False

        if get_longest_phrase_length(sentence)>100:
            continue
        if is_abnormal_text(sentence):
            continue
        record={"text":sentence}

        count+=1

        if count>max_count:
            file_id+=1
            count=0

        out_path = f"{out_dir}/model_{current_time_no_symbols}_{rand_id}_{file_id}.jsonl"
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                                                                    



# %%
question_list

# %%



