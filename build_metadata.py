"""Generate metadata (descriptions, precautions, symptom severities) for the
new 100-disease / 230-symptom dataset and APPEND to the existing CSVs.

Design / honesty notes:
- Descriptions are concise, factual, general-knowledge one-liners (same register
  as the rows already in symptom_Description.csv). They are NOT diagnoses.
- Precautions are conservative, generic self-care guidance and ALWAYS include
  consulting a healthcare professional. No specific drug/dosage advice.
- These auto-filled entries are placeholders for a research/educational demo and
  should be reviewed by a clinician before any real-world use.
- We append (dedup case-insensitively) so the original old-dataset rows survive.
"""
import csv, json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
schema = json.load(open("/tmp/new_schema.json"))
DISEASES = schema["diseases"]          # 100 names (lowercase as in dataset)
SYMPTOMS = schema["symptoms"]          # 230 column names

# ── Factual one-line descriptions (general medical knowledge) ────────────────
DESC = {
 "actinic keratosis":"A rough, scaly patch on the skin caused by years of sun exposure; considered precancerous.",
 "acute bronchiolitis":"Inflammation of the small airways (bronchioles) in the lungs, most common in infants and usually viral.",
 "acute bronchitis":"Short-term inflammation of the bronchial tubes causing cough and mucus, often following a respiratory infection.",
 "acute bronchospasm":"Sudden constriction of the airway muscles, causing wheezing and difficulty breathing.",
 "acute kidney injury":"A sudden decline in kidney function over hours or days, reducing the body's ability to filter waste.",
 "acute pancreatitis":"Sudden inflammation of the pancreas, typically causing severe upper-abdominal pain.",
 "acute sinusitis":"Short-term inflammation of the sinuses, usually from infection, causing facial pressure and congestion.",
 "allergy":"An immune-system overreaction to a normally harmless substance such as pollen, food, or dander.",
 "angina":"Chest pain caused by reduced blood flow to the heart muscle, often a warning sign of coronary artery disease.",
 "anxiety":"A mental-health condition marked by excessive worry, nervousness, and physical tension.",
 "appendicitis":"Inflammation of the appendix, causing abdominal pain that usually requires urgent surgery.",
 "arthritis of the hip":"Degeneration or inflammation of the hip joint, causing pain and reduced mobility.",
 "asthma":"A chronic condition in which the airways narrow and swell, producing wheezing and shortness of breath.",
 "benign prostatic hyperplasia (bph)":"Non-cancerous enlargement of the prostate gland that can obstruct urine flow in older men.",
 "brachial neuritis":"Sudden inflammation of nerves in the shoulder and arm, causing pain followed by weakness.",
 "bursitis":"Inflammation of a bursa (a fluid-filled cushioning sac) near a joint, causing localized pain.",
 "carpal tunnel syndrome":"Compression of the median nerve at the wrist, causing hand numbness, tingling, and weakness.",
 "cholecystitis":"Inflammation of the gallbladder, usually from a blocked bile duct, causing right-upper-abdominal pain.",
 "chronic back pain":"Persistent pain in the back lasting twelve weeks or longer, with many possible causes.",
 "chronic constipation":"Long-standing infrequent or difficult bowel movements.",
 "chronic obstructive pulmonary disease (copd)":"A progressive lung disease causing airflow obstruction and breathing difficulty, often from smoking.",
 "common cold":"A mild viral infection of the nose and throat causing congestion, sneezing, and sore throat.",
 "complex regional pain syndrome":"A chronic pain condition, usually affecting a limb, often following an injury.",
 "concussion":"A mild traumatic brain injury caused by a blow or jolt to the head.",
 "conjunctivitis":"Inflammation of the conjunctiva (the eye's outer membrane), causing redness and discharge.",
 "conjunctivitis due to allergy":"Eye inflammation triggered by an allergic reaction, causing itching, redness, and watering.",
 "contact dermatitis":"A skin rash caused by direct contact with an irritant or allergen.",
 "cornea infection":"Infection of the cornea (the clear front of the eye), which can threaten vision if untreated.",
 "croup":"A childhood viral infection causing airway swelling and a characteristic barking cough.",
 "cystitis":"Inflammation of the bladder, usually from a urinary tract infection, causing painful urination.",
 "degenerative disc disease":"Age-related breakdown of the spinal discs, which can cause back or neck pain.",
 "dental caries":"Tooth decay caused by bacteria producing acid that erodes the tooth enamel.",
 "depression":"A mood disorder causing persistent sadness, loss of interest, and reduced daily functioning.",
 "developmental disability":"A group of conditions causing impairment in physical, learning, language, or behavioral areas.",
 "diaper rash":"Skin irritation in the diaper area, common in infants, from moisture and friction.",
 "diverticulitis":"Inflammation or infection of small pouches (diverticula) in the wall of the colon.",
 "drug reaction":"An adverse or allergic response to a medication.",
 "ear drum damage":"A tear or rupture of the eardrum, which can affect hearing and cause ear pain.",
 "eczema":"A chronic condition causing itchy, inflamed, and dry patches of skin.",
 "esophagitis":"Inflammation of the esophagus, often from acid reflux, causing pain or difficulty swallowing.",
 "eustachian tube dysfunction (ear disorder)":"Impaired pressure regulation in the middle ear due to a blocked eustachian tube.",
 "fungal infection of the hair":"A fungal infection affecting the scalp or hair follicles.",
 "gallstone":"A hardened deposit in the gallbladder that can block bile flow and cause pain.",
 "gastrointestinal hemorrhage":"Bleeding anywhere along the digestive tract, ranging from minor to life-threatening.",
 "gout":"A form of arthritis caused by uric-acid crystal buildup, causing sudden severe joint pain.",
 "gum disease":"Inflammation and infection of the gums and supporting structures of the teeth.",
 "heart attack":"Death of heart muscle caused by a sudden blockage of blood flow to the heart; a medical emergency.",
 "heart failure":"A condition in which the heart cannot pump blood effectively to meet the body's needs.",
 "hemorrhoids":"Swollen veins in the lower rectum or anus, causing discomfort or bleeding.",
 "herniated disk":"A spinal disc whose inner material protrudes and presses on nearby nerves, causing pain.",
 "hiatal hernia":"A condition where part of the stomach pushes up through the diaphragm into the chest.",
 "hyperemesis gravidarum":"Severe, persistent nausea and vomiting during pregnancy that can cause dehydration.",
 "hypertensive heart disease":"Heart problems caused by chronic high blood pressure.",
 "hypoglycemia":"Abnormally low blood sugar, causing shakiness, confusion, and, if severe, loss of consciousness.",
 "idiopathic excessive menstruation":"Unusually heavy menstrual bleeding with no identifiable underlying cause.",
 "idiopathic irregular menstrual cycle":"Irregular menstrual periods with no identifiable underlying cause.",
 "idiopathic painful menstruation":"Painful periods (dysmenorrhea) with no identifiable underlying cause.",
 "infectious gastroenteritis":"Inflammation of the stomach and intestines from infection, causing diarrhea and vomiting.",
 "injury to the arm":"Physical trauma to the arm, such as a sprain, strain, or fracture.",
 "injury to the leg":"Physical trauma to the leg, such as a sprain, strain, or fracture.",
 "injury to the trunk":"Physical trauma to the torso, potentially affecting internal organs.",
 "liver disease":"Any condition that damages the liver and impairs its function.",
 "macular degeneration":"Progressive deterioration of the retina's macula, causing central-vision loss.",
 "marijuana abuse":"Problematic cannabis use that interferes with daily functioning or health.",
 "multiple sclerosis":"An autoimmune disease in which the immune system attacks the protective covering of nerves.",
 "noninfectious gastroenteritis":"Inflammation of the stomach and intestines not caused by infection (e.g., from diet or medication).",
 "nose disorder":"A general category of conditions affecting the nose, such as obstruction or bleeding.",
 "obstructive sleep apnea (osa)":"Repeated pauses in breathing during sleep caused by airway collapse.",
 "otitis externa (swimmer's ear)":"Infection of the outer ear canal, often from trapped moisture.",
 "otitis media":"Infection or inflammation of the middle ear, common in children.",
 "pain after an operation":"Post-surgical pain at or near the site of an operation.",
 "panic disorder":"An anxiety disorder marked by recurrent, unexpected panic attacks.",
 "pelvic inflammatory disease":"Infection of the female reproductive organs, often from sexually transmitted bacteria.",
 "peripheral nerve disorder":"Damage to nerves outside the brain and spinal cord, causing weakness, numbness, or pain.",
 "personality disorder":"A mental-health pattern of rigid, unhealthy thinking and behavior that impairs relationships.",
 "pneumonia":"Infection that inflames the air sacs of the lungs, which may fill with fluid.",
 "problem during pregnancy":"A general category of complications arising during pregnancy.",
 "psoriasis":"A chronic autoimmune skin condition causing thick, scaly, red patches.",
 "pyogenic skin infection":"A pus-producing bacterial infection of the skin.",
 "rectal disorder":"A general category of conditions affecting the rectum, such as bleeding or pain.",
 "schizophrenia":"A serious mental disorder affecting thinking, perception, emotions, and behavior.",
 "seasonal allergies (hay fever)":"Allergic inflammation of the nose and eyes triggered by seasonal allergens like pollen.",
 "sebaceous cyst":"A benign, fluid- or keratin-filled lump beneath the skin.",
 "sepsis":"A life-threatening, body-wide response to infection that can lead to organ failure.",
 "sickle cell crisis":"A painful episode in sickle cell disease when misshapen red blood cells block blood flow.",
 "sinus bradycardia":"A slower-than-normal heart rate originating from the heart's natural pacemaker.",
 "skin pigmentation disorder":"A condition causing abnormal lightening or darkening of the skin.",
 "skin polyp":"A small, usually benign growth projecting from the skin.",
 "spinal stenosis":"Narrowing of the spinal canal that puts pressure on the spinal cord or nerves.",
 "spondylosis":"Age-related wear of the spinal discs and joints, a form of spinal osteoarthritis.",
 "spontaneous abortion":"The natural loss of a pregnancy before the 20th week (miscarriage).",
 "sprain or strain":"Injury to a ligament (sprain) or to a muscle/tendon (strain).",
 "strep throat":"A bacterial throat infection caused by group A streptococcus, causing severe sore throat.",
 "stye":"A painful red lump on the eyelid caused by an infected oil gland.",
 "temporary or benign blood in urine":"The presence of blood in the urine that is transient or from a non-serious cause.",
 "threatened pregnancy":"Early-pregnancy bleeding that raises the risk of miscarriage but where the pregnancy may continue.",
 "urinary tract infection":"A bacterial infection anywhere in the urinary system, most often the bladder.",
 "vaginal cyst":"A benign fluid-filled lump in the vaginal wall.",
 "vaginitis":"Inflammation of the vagina, often causing discharge, itching, and discomfort.",
 "vulvodynia":"Chronic pain or discomfort around the opening of the vagina with no identifiable cause.",
}

# ── Precaution templates by clinical group (conservative, generic) ───────────
GEN = ["consult a healthcare professional","follow prescribed treatment","monitor your symptoms","rest and stay hydrated"]
P = {
 "respiratory":["consult a healthcare professional","avoid smoke and known irritants","stay hydrated","rest and monitor breathing"],
 "cardiac":["seek prompt medical evaluation","avoid strenuous exertion until cleared","follow prescribed medication","monitor for worsening symptoms"],
 "gi":["consult a healthcare professional","stay hydrated","eat bland easy-to-digest food","avoid trigger foods"],
 "gu":["consult a healthcare professional","drink plenty of water","maintain good hygiene","complete any prescribed treatment"],
 "gyn":["consult a healthcare professional","track your symptoms and cycle","maintain good hygiene","seek care if pain is severe"],
 "preg":["seek prompt obstetric care","avoid strenuous activity","stay hydrated","attend follow-up appointments"],
 "msk":["rest the affected area","apply ice and elevate if swollen","avoid aggravating activity","consult a healthcare professional"],
 "neuro":["seek medical evaluation","avoid driving until assessed","rest and avoid overexertion","follow up with a specialist"],
 "psych":["consult a mental-health professional","maintain a support network","keep a regular routine","seek urgent help in crisis"],
 "derm":["keep the area clean and dry","avoid scratching or irritants","consult a healthcare professional","use protective skin care"],
 "eye":["avoid touching or rubbing the eye","maintain eye hygiene","consult an eye-care professional","seek care if vision changes"],
 "ent":["consult a healthcare professional","keep the ear/mouth area clean","avoid inserting objects","complete any prescribed treatment"],
 "trauma":["seek medical evaluation","immobilize and rest the area","apply ice to reduce swelling","follow up as advised"],
 "systemic":["seek prompt medical care","monitor symptoms closely","stay hydrated","follow prescribed treatment"],
}
# disease -> precaution group
GRP = {
 "actinic keratosis":"derm","acute bronchiolitis":"respiratory","acute bronchitis":"respiratory","acute bronchospasm":"respiratory",
 "acute kidney injury":"gu","acute pancreatitis":"gi","acute sinusitis":"respiratory","allergy":"systemic","angina":"cardiac",
 "anxiety":"psych","appendicitis":"gi","arthritis of the hip":"msk","asthma":"respiratory","benign prostatic hyperplasia (bph)":"gu",
 "brachial neuritis":"neuro","bursitis":"msk","carpal tunnel syndrome":"msk","cholecystitis":"gi","chronic back pain":"msk",
 "chronic constipation":"gi","chronic obstructive pulmonary disease (copd)":"respiratory","common cold":"respiratory",
 "complex regional pain syndrome":"neuro","concussion":"neuro","conjunctivitis":"eye","conjunctivitis due to allergy":"eye",
 "contact dermatitis":"derm","cornea infection":"eye","croup":"respiratory","cystitis":"gu","degenerative disc disease":"msk",
 "dental caries":"ent","depression":"psych","developmental disability":"psych","diaper rash":"derm","diverticulitis":"gi",
 "drug reaction":"derm","ear drum damage":"ent","eczema":"derm","esophagitis":"gi","eustachian tube dysfunction (ear disorder)":"ent",
 "fungal infection of the hair":"derm","gallstone":"gi","gastrointestinal hemorrhage":"gi","gout":"msk","gum disease":"ent",
 "heart attack":"cardiac","heart failure":"cardiac","hemorrhoids":"gi","herniated disk":"msk","hiatal hernia":"gi",
 "hyperemesis gravidarum":"preg","hypertensive heart disease":"cardiac","hypoglycemia":"systemic",
 "idiopathic excessive menstruation":"gyn","idiopathic irregular menstrual cycle":"gyn","idiopathic painful menstruation":"gyn",
 "infectious gastroenteritis":"gi","injury to the arm":"trauma","injury to the leg":"trauma","injury to the trunk":"trauma",
 "liver disease":"gi","macular degeneration":"eye","marijuana abuse":"psych","multiple sclerosis":"neuro",
 "noninfectious gastroenteritis":"gi","nose disorder":"ent","obstructive sleep apnea (osa)":"respiratory",
 "otitis externa (swimmer's ear)":"ent","otitis media":"ent","pain after an operation":"trauma","panic disorder":"psych",
 "pelvic inflammatory disease":"gyn","peripheral nerve disorder":"neuro","personality disorder":"psych","pneumonia":"respiratory",
 "problem during pregnancy":"preg","psoriasis":"derm","pyogenic skin infection":"derm","rectal disorder":"gi",
 "schizophrenia":"psych","seasonal allergies (hay fever)":"respiratory","sebaceous cyst":"derm","sepsis":"systemic",
 "sickle cell crisis":"systemic","sinus bradycardia":"cardiac","skin pigmentation disorder":"derm","skin polyp":"derm",
 "spinal stenosis":"msk","spondylosis":"msk","spontaneous abortion":"preg","sprain or strain":"msk","strep throat":"respiratory",
 "stye":"eye","temporary or benign blood in urine":"gu","threatened pregnancy":"preg","urinary tract infection":"gu",
 "vaginal cyst":"gyn","vaginitis":"gyn","vulvodynia":"gyn",
}

def title(d):
    return d  # keep dataset's own casing for exact match in predict.py

# ── Append helpers (dedup case-insensitively against existing rows) ──────────
def existing_keys(path, col0=0):
    keys=set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            r=csv.reader(f)
            next(r, None)
            for row in r:
                if row: keys.add(row[col0].strip().lower())
    return keys

# Descriptions
desc_path=os.path.join(ROOT,"symptom_Description.csv")
have=existing_keys(desc_path)
added=0
with open(desc_path,"a",newline="") as f:
    w=csv.writer(f)
    for d in DISEASES:
        if d.lower() in have: continue
        w.writerow([title(d), DESC.get(d, f"{d.capitalize()} — a medical condition. Auto-generated placeholder; consult a clinician for details.")])
        added+=1
print(f"descriptions appended: {added}")

# Precautions
prec_path=os.path.join(ROOT,"symptom_precaution.csv")
have=existing_keys(prec_path)
added=0
with open(prec_path,"a",newline="") as f:
    w=csv.writer(f)
    for d in DISEASES:
        if d.lower() in have: continue
        precs=P.get(GRP.get(d,""), GEN)
        w.writerow([title(d)]+precs[:4])
        added+=1
print(f"precautions appended: {added}")

# ── Severity weights for the 230 NEW symptom names ───────────────────────────
# Heuristic 1-7 tiers (matches the old file's 1-7 range). predict.py defaults
# unknown symptoms to 0, so this only adds signal, never breaks anything.
HIGH=["seizures","fainting","vomiting blood","hemoptysis","coughing up sputum","melena","rectal bleeding",
 "blood in stool","bleeding from eye","blindness","apnea","difficulty breathing","shortness of breath",
 "sharp chest pain","burning chest pain","chest tightness","irregular heartbeat","delusions or hallucinations",
 "loss of sensation","focal weakness","paresthesia","jaundice","blood in urine","retention of urine",
 "low urine output","spotting or bleeding during pregnancy","uterine contractions","palpitations",
 "decreased heart rate","increased heart rate","abnormal involuntary movements","difficulty speaking",
 "symptoms of the kidneys","kidney mass","sharp abdominal pain"]
MED=["fever","chills","vomiting","diarrhea","nausea","headache","dizziness","cough","wheezing","sore throat",
 "difficulty in swallowing","regurgitation","regurgitation.1","burning abdominal pain","upper abdominal pain",
 "lower abdominal pain","side pain","painful urination","frequent urination","involuntary urination",
 "excessive urination at night","pelvic pain","pain during pregnancy","heavy menstrual flow","long menstrual periods",
 "painful menstruation","depression","depressive or psychotic symptoms","anxiety and nervousness","insomnia",
 "weakness","ache all over","joint pain","back pain","neck pain","low back pain","knee pain","hip pain",
 "ear pain","pain in eye","double vision","diminished vision","diminished hearing","peripheral edema",
 "fluid retention","sweating","feeling ill","fatigue","groin pain","rib pain","abnormal breathing sounds",
 "hurts to breath","congestion in chest","sinus congestion","facial pain","toothache","mouth ulcer",
 "nosebleed","bleeding gums","pain in gums","swollen or red tonsils","skin lesion","abnormal appearing skin",
 "skin swelling","lip swelling","allergic reaction","seizures" ]
def weight(sym):
    if sym in HIGH: return 6
    if sym in MED: return 4
    # everything else: mild signal
    return 2
sev_path=os.path.join(ROOT,"Symptom-severity.csv")
have=existing_keys(sev_path)
added=0
with open(sev_path,"a",newline="") as f:
    w=csv.writer(f)
    for s in SYMPTOMS:
        if s.lower() in have: continue
        w.writerow([s, weight(s)])
        added+=1
print(f"severity rows appended: {added}")

# ── Verify coverage ──────────────────────────────────────────────────────────
d_keys=existing_keys(desc_path); p_keys=existing_keys(prec_path); s_keys=existing_keys(sev_path)
miss_d=[d for d in DISEASES if d.lower() not in d_keys]
miss_p=[d for d in DISEASES if d.lower() not in p_keys]
miss_s=[s for s in SYMPTOMS if s.lower() not in s_keys]
print("\nCOVERAGE CHECK")
print(" diseases w/o description:", miss_d or "NONE")
print(" diseases w/o precautions:", miss_p or "NONE")
print(" symptoms w/o severity   :", miss_s or "NONE")
