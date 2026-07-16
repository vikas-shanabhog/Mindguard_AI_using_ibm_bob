"""
MindGuard AI – Early Distress Detection Assistant
==================================================
An Agentic AI-powered early emotional distress detection web application
built with Python, Flask, IBM watsonx.ai Granite Models, and a RAG pipeline
for mental health awareness and coping strategies.

Disclaimer: This AI assistant provides educational and emotional support only.
It is not a substitute for professional medical advice, diagnosis, or treatment.
"""

import os
import json
import re
import textwrap
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "mindguard-ai-secret-2024")

# IBM watsonx.ai Configuration – set via environment variables
WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY", "2f_HxjEGmMQopogbgpOEUtSBprds9kC45n35-Bb06Vna")
WATSONX_URL     = os.environ.get("WATSONX_URL", " https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "12f032c6-400d-4253-b52a-40dcacf7a7d6")
GRANITE_MODEL_ID   = os.environ.get("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2")

# ---------------------------------------------------------------------------
# RAG Knowledge Base – Mental Health Awareness Resources
# ---------------------------------------------------------------------------

RAG_KNOWLEDGE_BASE = [
    {
        "id": "kb_001",
        "topic": "Stress Management",
        "keywords": ["stress", "overwhelmed", "pressure", "workload", "deadlines"],
        "content": (
            "Stress is the body's natural response to challenges. Effective stress management "
            "techniques include time-blocking your schedule, practicing deep-breathing exercises "
            "(4-7-8 technique), progressive muscle relaxation, regular physical exercise, and "
            "setting healthy boundaries. Breaking tasks into smaller steps and prioritising them "
            "using the Eisenhower Matrix can also reduce overwhelm significantly."
        ),
        "resource": "American Psychological Association – Stress Management Guide",
        "link": "https://www.apa.org/topics/stress",
    },
    {
        "id": "kb_002",
        "topic": "Anxiety Awareness",
        "keywords": ["anxiety", "anxious", "worry", "fear", "panic", "nervous", "restless"],
        "content": (
            "Anxiety disorders are among the most common mental health conditions. Cognitive "
            "Behavioural Therapy (CBT) is highly effective. Grounding techniques such as the "
            "5-4-3-2-1 method help anchor you to the present moment. Mindfulness meditation, "
            "limiting caffeine intake, consistent sleep schedules, and journalling can all "
            "reduce anxiety symptoms over time."
        ),
        "resource": "Anxiety & Depression Association of America (ADAA)",
        "link": "https://adaa.org/understanding-anxiety",
    },
    {
        "id": "kb_003",
        "topic": "Sadness & Depression Awareness",
        "keywords": ["sad", "sadness", "depressed", "hopeless", "empty", "worthless", "meaningless", "crying"],
        "content": (
            "Persistent sadness may indicate depression, a treatable medical condition. Key "
            "approaches include behavioural activation (engaging in enjoyable activities even "
            "when motivation is low), establishing a daily routine, social connection, sunlight "
            "exposure, and physical movement. Professional evaluation is important when symptoms "
            "persist for more than two weeks."
        ),
        "resource": "National Institute of Mental Health – Depression",
        "link": "https://www.nimh.nih.gov/health/topics/depression",
    },
    {
        "id": "kb_004",
        "topic": "Loneliness & Social Isolation",
        "keywords": ["lonely", "loneliness", "isolated", "alone", "disconnected", "no one", "nobody"],
        "content": (
            "Loneliness is a universal human experience that can significantly impact mental and "
            "physical health. Strategies include reaching out to one person daily, joining "
            "community groups or clubs, volunteering, practising self-compassion, and using "
            "structured social activities to gradually rebuild connection. Online support "
            "communities can also be a valuable starting point."
        ),
        "resource": "Mind UK – How to Cope with Loneliness",
        "link": "https://www.mind.org.uk/information-support/tips-for-everyday-living/loneliness/",
    },
    {
        "id": "kb_005",
        "topic": "Burnout & Emotional Exhaustion",
        "keywords": ["burnout", "exhausted", "drained", "tired", "no energy", "motivation", "empty", "numb"],
        "content": (
            "Burnout is a state of chronic stress leading to physical and emotional exhaustion. "
            "Recovery involves setting firm work-life boundaries, taking restorative breaks, "
            "practising the STOP technique (Stop, Take a breath, Observe, Proceed), reconnecting "
            "with hobbies, delegating tasks, and seeking peer support. Adequate sleep (7-9 hours) "
            "is foundational to recovery."
        ),
        "resource": "Mayo Clinic – Job Burnout: How to Spot It and Take Action",
        "link": "https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/burnout/art-20046642",
    },
    {
        "id": "kb_006",
        "topic": "Mindfulness & Meditation",
        "keywords": ["mindfulness", "meditation", "calm", "relaxation", "breathing", "present"],
        "content": (
            "Mindfulness-Based Stress Reduction (MBSR) is an evidence-based programme that "
            "teaches present-moment awareness. Even 10 minutes of daily meditation using apps "
            "like Calm or Headspace can reduce cortisol levels. Body scan meditations, loving-"
            "kindness meditation, and mindful walking are excellent starting practices that "
            "require no equipment."
        ),
        "resource": "Mindful.org – Getting Started with Mindfulness",
        "link": "https://www.mindful.org/meditation/mindfulness-getting-started/",
    },
    {
        "id": "kb_007",
        "topic": "Crisis & High-Risk Support",
        "keywords": ["suicide", "suicidal", "end my life", "kill myself", "self-harm", "hurt myself",
                     "don't want to live", "no reason to live", "give up", "disappear"],
        "content": (
            "If you are experiencing a mental health crisis or thoughts of self-harm, please "
            "reach out for immediate support. You are not alone and help is available. "
            "Contact a trusted person, a licensed mental health professional, or your local "
            "emergency services immediately. Crisis helplines are available 24/7."
        ),
        "resource": "International Association for Suicide Prevention – Crisis Centres",
        "link": "https://www.iasp.info/resources/Crisis_Centres/",
        "crisis": True,
    },
    {
        "id": "kb_008",
        "topic": "Sleep & Mental Health",
        "keywords": ["sleep", "insomnia", "can't sleep", "tired", "restless", "nightmares"],
        "content": (
            "Sleep and mental health are deeply interconnected. Poor sleep worsens anxiety, "
            "depression, and stress. Sleep hygiene practices include maintaining consistent "
            "sleep/wake times, avoiding screens 1 hour before bed, keeping the bedroom cool "
            "and dark, avoiding caffeine after 2 PM, and using relaxation techniques such as "
            "progressive muscle relaxation or white noise."
        ),
        "resource": "Sleep Foundation – Mental Health and Sleep",
        "link": "https://www.sleepfoundation.org/mental-health",
    },
]

# ---------------------------------------------------------------------------
# RAG Retrieval Function
# ---------------------------------------------------------------------------

def retrieve_rag_context(text: str, top_k: int = 3) -> list[dict]:
    """
    Simple keyword-overlap RAG retrieval from the knowledge base.
    Scores each KB entry by how many of its keywords appear in the user text.
    Returns the top_k most relevant entries.
    """
    text_lower = text.lower()
    scored = []
    for entry in RAG_KNOWLEDGE_BASE:
        score = sum(1 for kw in entry["keywords"] if kw in text_lower)
        if score > 0:
            scored.append((score, entry))

    # Always include crisis entry if any crisis keyword is present
    crisis_entry = next((e for e in RAG_KNOWLEDGE_BASE if e.get("crisis")), None)
    is_crisis = crisis_entry and any(kw in text_lower for kw in crisis_entry["keywords"])

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [entry for _, entry in scored[:top_k]]

    # Ensure crisis entry is always first if triggered
    if is_crisis and crisis_entry and crisis_entry not in results:
        results.insert(0, crisis_entry)

    return results, is_crisis

# ---------------------------------------------------------------------------
# IBM watsonx.ai Granite Model Interface
# ---------------------------------------------------------------------------

def get_watsonx_client():
    """Initialise IBM watsonx.ai API client."""
    credentials = Credentials(
        url=WATSONX_URL,
        api_key=WATSONX_API_KEY,
    )
    return APIClient(credentials)

def build_agent_prompt(journal_text: str, rag_context: list[dict], is_crisis: bool) -> str:
    """
    Build the structured agentic prompt for the Early Distress Detection Agent.
    Injects retrieved RAG context for grounded, evidence-based responses.
    """
    rag_snippets = "\n\n".join(
        f"[Resource {i+1}] Topic: {e['topic']}\n{e['content']}"
        for i, e in enumerate(rag_context)
    )

    crisis_instruction = (
        "\n⚠️  IMPORTANT: The journal entry contains high-risk language. "
        "You MUST immediately and empathetically acknowledge this, strongly encourage the user "
        "to contact a trusted person, a licensed mental health professional, or local emergency "
        "services (e.g. 988 Suicide & Crisis Lifeline in the US, or their local equivalent). "
        "Lead with compassion and urgent safety guidance.\n"
        if is_crisis else ""
    )

    prompt = textwrap.dedent(f"""
    You are MindGuard AI, an empathetic Early Distress Detection Assistant powered by IBM Granite.
    Your role is to analyse the user's journal entry for signs of emotional distress, provide
    compassionate support, and offer evidence-based coping strategies.
    {crisis_instruction}
    === MENTAL HEALTH KNOWLEDGE BASE (Retrieved Context) ===
    {rag_snippets if rag_snippets else "No specific resources retrieved. Use general mental health best practices."}
    =========================================================

    === USER JOURNAL ENTRY ===
    {journal_text}
    ==========================

    Analyse the journal entry above and respond using EXACTLY this JSON structure:

    {{
      "detected_emotions": ["<emotion1>", "<emotion2>", ...],
      "distress_level": "<Low|Moderate|High>",
      "distress_score": <integer 1-10>,
      "emotional_summary": "<2-3 sentence empathetic summary of the emotional state>",
      "insights": ["<insight1>", "<insight2>", "<insight3>"],
      "coping_strategies": [
        {{"title": "<strategy name>", "description": "<practical description>"}},
        {{"title": "<strategy name>", "description": "<practical description>"}},
        {{"title": "<strategy name>", "description": "<practical description>"}}
      ],
      "self_care_activities": ["<activity1>", "<activity2>", "<activity3>"],
      "affirmation": "<a short, warm, personalised affirmation for the user>",
      "seek_professional_help": <true|false>,
      "crisis_response": "<crisis guidance if high-risk language detected, else null>",
      "rag_resources": [
        {{"topic": "<topic>", "resource": "<resource name>", "link": "<url>"}}
      ]
    }}

    Respond ONLY with valid JSON. Do not include any text outside the JSON object.
    """).strip()

    return prompt


def analyse_with_granite(journal_text: str) -> dict:
    """
    Core agentic pipeline:
    1. Retrieve relevant RAG context from the knowledge base.
    2. Build structured prompt with injected context.
    3. Call IBM Granite model for AI-powered emotional analysis.
    4. Parse and return structured JSON response.
    """
    rag_context, is_crisis = retrieve_rag_context(journal_text)
    prompt = build_agent_prompt(journal_text, rag_context, is_crisis)

    try:
        client = get_watsonx_client()
        model = ModelInference(
            model_id=GRANITE_MODEL_ID,
            api_client=client,
            project_id=WATSONX_PROJECT_ID,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 1200,
                GenParams.MIN_NEW_TOKENS: 100,
                GenParams.TEMPERATURE: 0.3,
                GenParams.REPETITION_PENALTY: 1.1,
                GenParams.STOP_SEQUENCES: ["```", "==="],
            },
        )
        response = model.generate_text(prompt=prompt)
        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError("No JSON object found in model response.")

    except Exception as exc:
        # Fallback: return a well-structured mock response for demo/dev mode
        app.logger.warning(f"watsonx.ai call failed ({exc}). Returning demo response.")
        result = _demo_fallback(journal_text, rag_context, is_crisis)

    # Always inject retrieved RAG resources
    result["rag_resources"] = [
        {"topic": e["topic"], "resource": e["resource"], "link": e["link"]}
        for e in rag_context
    ]
    result["is_crisis"] = is_crisis
    result["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    result["model"] = GRANITE_MODEL_ID
    return result


def _demo_fallback(journal_text: str, rag_context: list[dict], is_crisis: bool) -> dict:
    """
    Demo fallback used when watsonx.ai credentials are not configured.
    Provides a realistic illustrative response so the UI is fully demonstrable.
    """
    text_lower = journal_text.lower()

    emotions = []
    if any(w in text_lower for w in ["anxious", "anxiety", "nervous", "worry", "panic"]):
        emotions.append("Anxiety")
    if any(w in text_lower for w in ["stress", "overwhelmed", "pressure"]):
        emotions.append("Stress")
    if any(w in text_lower for w in ["sad", "sadness", "depressed", "hopeless", "empty"]):
        emotions.append("Sadness")
    if any(w in text_lower for w in ["lonely", "alone", "isolated", "disconnected"]):
        emotions.append("Loneliness")
    if any(w in text_lower for w in ["burnout", "exhausted", "drained", "tired", "numb"]):
        emotions.append("Burnout / Exhaustion")
    if not emotions:
        emotions = ["Mild Emotional Strain"]

    score = min(10, 3 + len(emotions) * 2 + (3 if is_crisis else 0))
    level = "High" if score >= 8 or is_crisis else ("Moderate" if score >= 5 else "Low")

    return {
        "detected_emotions": emotions,
        "distress_level": level,
        "distress_score": score,
        "emotional_summary": (
            "Your journal entry reflects genuine emotional weight that deserves acknowledgment. "
            "What you're feeling is valid, and recognising these emotions is already a meaningful "
            "step toward well-being. You are not alone in this experience."
        ),
        "insights": [
            "Your words suggest you may be carrying more than you realise — it's okay to pause and rest.",
            "Emotions like these are signals, not weaknesses. They guide us toward what needs attention.",
            "Small, consistent acts of self-care can create meaningful shifts in how you feel day to day.",
        ],
        "coping_strategies": [
            {
                "title": "Mindful Breathing (4-7-8 Technique)",
                "description": "Inhale for 4 seconds, hold for 7 seconds, exhale slowly for 8 seconds. "
                               "Repeat 3–4 times to activate your parasympathetic nervous system.",
            },
            {
                "title": "Grounding – 5-4-3-2-1 Method",
                "description": "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, "
                               "and 1 you taste. This anchors you to the present moment.",
            },
            {
                "title": "Journalling with Intention",
                "description": "Continue writing, but try shifting toward gratitude or "
                               "'what went well today' prompts to gently re-frame your perspective.",
            },
        ],
        "self_care_activities": [
            "A 20-minute walk in natural light",
            "Prepare and eat a nourishing meal mindfully",
            "Reach out to one trusted person today",
        ],
        "affirmation": (
            "You are showing great courage by reflecting on your feelings. "
            "Every step you take toward understanding yourself matters."
        ),
        "seek_professional_help": level in ("Moderate", "High"),
        "crisis_response": (
            "⚠️ Your message contains language that concerns us deeply. Please know that you matter "
            "and help is available right now. Please contact the 988 Suicide & Crisis Lifeline "
            "(call or text 988 in the US), or reach out to your local emergency services immediately. "
            "You are not alone."
            if is_crisis else None
        ),
    }


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main MindGuard AI single-page application."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """
    POST /api/analyse
    Body: { "journal_text": "..." }
    Returns: JSON emotional analysis result from the Granite-powered agent.
    """
    data = request.get_json(force=True, silent=True) or {}
    journal_text = (data.get("journal_text") or "").strip()

    if not journal_text:
        return jsonify({"error": "Please provide a journal entry to analyse."}), 400
    if len(journal_text) < 10:
        return jsonify({"error": "Journal entry is too short. Please share more details."}), 400
    if len(journal_text) > 3000:
        return jsonify({"error": "Journal entry is too long. Please limit to 3000 characters."}), 400

    result = analyse_with_granite(journal_text)
    return jsonify(result)


@app.route("/api/resources", methods=["GET"])
def api_resources():
    """GET /api/resources – Return all knowledge base topics."""
    resources = [
        {"topic": e["topic"], "resource": e["resource"], "link": e["link"]}
        for e in RAG_KNOWLEDGE_BASE
        if not e.get("crisis")
    ]
    return jsonify(resources)


@app.route("/api/health", methods=["GET"])
def api_health():
    """GET /api/health – Application health check."""
    return jsonify({
        "status": "ok",
        "app": "MindGuard AI",
        "model": GRANITE_MODEL_ID,
        "timestamp": datetime.utcnow().isoformat(),
    })


# ---------------------------------------------------------------------------
# HTML / CSS / JS Template (Single-Page Application)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MindGuard AI – Early Distress Detection Assistant</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" />
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" />
  <style>
    :root {
      --ibm-blue:      #0f62fe;
      --ibm-blue-dark: #0043ce;
      --ibm-teal:      #009d9a;
      --ibm-purple:    #8a3ffc;
      --ibm-red:       #da1e28;
      --ibm-green:     #24a148;
      --ibm-yellow:    #f1c21b;
      --bg:            #f4f4f4;
      --card-bg:       #ffffff;
      --text:          #161616;
      --muted:         #6f6f6f;
      --border:        #e0e0e0;
      --low-color:     #24a148;
      --mod-color:     #f1c21b;
      --high-color:    #da1e28;
    }

    * { box-sizing: border-box; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      font-size: 15px;
      line-height: 1.6;
    }

    /* ── Navbar ── */
    .navbar-brand .brand-icon { font-size: 1.5rem; color: var(--ibm-teal); }
    .navbar-brand .brand-name { font-weight: 700; font-size: 1.15rem; letter-spacing: -0.3px; }
    .navbar-brand .brand-sub  { font-size: 0.7rem; color: var(--muted); display: block; line-height: 1; }

    /* ── Hero ── */
    .hero {
      background: linear-gradient(135deg, #001141 0%, #0f62fe 60%, #009d9a 100%);
      color: #fff;
      padding: 3.5rem 1rem 2.5rem;
      text-align: center;
    }
    .hero h1 { font-size: clamp(1.6rem, 4vw, 2.4rem); font-weight: 700; margin-bottom: .5rem; }
    .hero p  { font-size: 1rem; opacity: .85; max-width: 640px; margin: 0 auto .6rem; }
    .hero .badge-ibm {
      display: inline-flex; align-items: center; gap: .4rem;
      background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.3);
      border-radius: 20px; padding: .25rem .75rem; font-size: .75rem; margin-top: .5rem;
    }

    /* ── Disclaimer ── */
    .disclaimer {
      background: #fff8e1; border-left: 4px solid var(--ibm-yellow);
      border-radius: 4px; padding: .75rem 1rem; font-size: .82rem;
      color: #5a4700; margin-bottom: 1.5rem;
    }

    /* ── Journal card ── */
    .journal-card { border: none; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08); }
    .journal-card .card-header {
      background: var(--ibm-blue); color: #fff;
      border-radius: 12px 12px 0 0 !important; font-weight: 600;
    }
    #journalText {
      border: 2px solid var(--border); border-radius: 8px;
      font-size: .95rem; resize: vertical; min-height: 160px;
      transition: border-color .2s;
    }
    #journalText:focus { border-color: var(--ibm-blue); box-shadow: 0 0 0 3px rgba(15,98,254,.15); outline: none; }

    .char-count { font-size: .78rem; color: var(--muted); text-align: right; }
    .char-count.warn { color: var(--ibm-red); }

    .btn-analyse {
      background: var(--ibm-blue); color: #fff; border: none;
      padding: .65rem 2rem; border-radius: 8px; font-weight: 600; font-size: 1rem;
      transition: background .2s, transform .1s;
    }
    .btn-analyse:hover { background: var(--ibm-blue-dark); color: #fff; }
    .btn-analyse:active { transform: scale(.98); }
    .btn-analyse:disabled { opacity: .6; cursor: not-allowed; }

    /* ── Prompt chips ── */
    .prompt-chip {
      display: inline-block; cursor: pointer;
      background: #e8f0fe; border: 1px solid #bcd0fb; border-radius: 20px;
      padding: .2rem .7rem; font-size: .78rem; color: var(--ibm-blue);
      margin: .2rem; transition: background .15s;
    }
    .prompt-chip:hover { background: #d0e2ff; }

    /* ── Loader ── */
    .ai-loader {
      display: none; flex-direction: column; align-items: center;
      gap: 1rem; padding: 2.5rem 1rem; text-align: center;
    }
    .ai-loader.show { display: flex; }
    .loader-ring {
      width: 56px; height: 56px; border-radius: 50%;
      border: 5px solid #e0e0e0; border-top-color: var(--ibm-blue);
      animation: spin .9s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loader-steps { font-size: .83rem; color: var(--muted); }

    /* ── Results ── */
    #resultsSection { display: none; }
    #resultsSection.show { display: block; animation: fadeIn .4s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

    /* ── Crisis banner ── */
    .crisis-banner {
      background: #fff1f1; border: 2px solid var(--ibm-red);
      border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
    }
    .crisis-banner h5 { color: var(--ibm-red); font-weight: 700; }

    /* ── Distress level indicator ── */
    .distress-card { border-radius: 12px; overflow: hidden; }
    .distress-header {
      padding: 1.2rem 1.5rem; color: #fff; display: flex;
      align-items: center; justify-content: space-between;
    }
    .distress-header.low    { background: var(--low-color); }
    .distress-header.moderate { background: #c17f24; }
    .distress-header.high   { background: var(--ibm-red); }
    .distress-score-circle {
      width: 64px; height: 64px; border-radius: 50%;
      background: rgba(255,255,255,.2); border: 3px solid rgba(255,255,255,.6);
      display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .distress-score-circle .score-num { font-size: 1.5rem; font-weight: 800; line-height: 1; }
    .distress-score-circle .score-lbl { font-size: .6rem; opacity: .85; }

    /* ── Emotion tags ── */
    .emotion-tag {
      display: inline-flex; align-items: center; gap: .3rem;
      background: #e8eaf6; color: #283593; border-radius: 20px;
      padding: .25rem .75rem; font-size: .8rem; margin: .2rem; font-weight: 600;
    }

    /* ── Section cards ── */
    .result-card { border: none; border-radius: 12px; box-shadow: 0 1px 8px rgba(0,0,0,.07); margin-bottom: 1.2rem; }
    .result-card .card-header {
      background: #f4f6ff; border-bottom: 1px solid #dde3f5;
      font-weight: 700; font-size: .9rem; color: var(--ibm-blue-dark);
      border-radius: 12px 12px 0 0 !important;
    }
    .result-card .card-header i { margin-right: .4rem; }

    /* ── Coping strategy cards ── */
    .strategy-card {
      background: #f0fdf4; border-left: 4px solid var(--ibm-green);
      border-radius: 8px; padding: .9rem 1rem; margin-bottom: .8rem;
    }
    .strategy-card h6 { color: #166534; font-weight: 700; margin-bottom: .25rem; }
    .strategy-card p  { font-size: .85rem; color: #1e4d2b; margin: 0; }

    /* ── Self-care list ── */
    .self-care-item {
      display: flex; align-items: center; gap: .6rem;
      padding: .5rem 0; border-bottom: 1px solid #f0f0f0; font-size: .88rem;
    }
    .self-care-item:last-child { border: none; }
    .self-care-item i { color: var(--ibm-teal); font-size: 1rem; }

    /* ── Affirmation box ── */
    .affirmation-box {
      background: linear-gradient(135deg, #f5f0ff 0%, #e8f4fd 100%);
      border-left: 5px solid var(--ibm-purple); border-radius: 10px;
      padding: 1.2rem 1.5rem; font-style: italic; font-size: .97rem;
      color: #3b1a7a; margin-bottom: 1.2rem;
    }

    /* ── RAG resources ── */
    .rag-resource {
      display: flex; align-items: center; gap: .8rem;
      padding: .6rem; border-radius: 8px; background: #f9fafb;
      border: 1px solid var(--border); margin-bottom: .6rem;
    }
    .rag-resource i { color: var(--ibm-blue); font-size: 1.1rem; }
    .rag-resource a { color: var(--ibm-blue); font-weight: 600; font-size: .85rem; text-decoration: none; }
    .rag-resource a:hover { text-decoration: underline; }

    /* ── Professional help ── */
    .prof-help-box {
      background: #fff7ed; border: 1px solid #fed7aa; border-radius: 10px;
      padding: 1rem 1.25rem; margin-bottom: 1.2rem;
    }
    .prof-help-box h6 { color: #9a3412; font-weight: 700; }

    /* ── Insight list ── */
    .insight-item { display: flex; gap: .6rem; margin-bottom: .6rem; font-size: .88rem; }
    .insight-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--ibm-blue); flex-shrink: 0; margin-top: .45rem;
    }

    /* ── Meta bar ── */
    .meta-bar {
      background: #f4f4f4; border-top: 1px solid var(--border);
      font-size: .75rem; color: var(--muted); padding: .5rem 1rem;
      border-radius: 0 0 12px 12px; display: flex; gap: 1.5rem; flex-wrap: wrap;
    }
    .meta-bar span i { margin-right: .2rem; }

    /* ── Resources tab ── */
    .resource-topic-card {
      border: none; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,.07);
      margin-bottom: 1rem; overflow: hidden;
    }
    .resource-topic-card .rtc-header {
      background: var(--ibm-blue); color: #fff; padding: .65rem 1rem;
      font-weight: 600; font-size: .9rem;
    }
    .resource-topic-card .rtc-body { padding: .9rem 1rem; font-size: .86rem; }

    /* ── Footer ── */
    footer {
      text-align: center; padding: 2rem 1rem 1.5rem;
      font-size: .78rem; color: var(--muted); border-top: 1px solid var(--border);
      margin-top: 3rem;
    }

    /* ── Responsive tweaks ── */
    @media (max-width: 576px) {
      .hero { padding: 2.5rem 1rem 2rem; }
      .distress-header { flex-direction: column; gap: .8rem; text-align: center; }
    }
  </style>
</head>
<body>

<!-- ═══════════════════════ NAVBAR ═══════════════════════ -->
<nav class="navbar navbar-expand-lg navbar-dark bg-dark px-3 py-2">
  <a class="navbar-brand d-flex align-items-center gap-2" href="#">
    <i class="bi bi-shield-heart brand-icon"></i>
    <div>
      <span class="brand-name">MindGuard AI</span>
      <span class="brand-sub">Early Distress Detection Assistant</span>
    </div>
  </a>
  <button class="navbar-toggler" type="button" data-bs-toggle="collapse"
          data-bs-target="#navMenu" aria-controls="navMenu" aria-expanded="false">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navMenu">
    <ul class="navbar-nav ms-auto align-items-lg-center gap-lg-1">
      <li class="nav-item">
        <a class="nav-link" href="#analyseSection"><i class="bi bi-journal-text me-1"></i>Analyse</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#resourcesSection"><i class="bi bi-book me-1"></i>Resources</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#aboutSection"><i class="bi bi-info-circle me-1"></i>About</a>
      </li>
      <li class="nav-item ms-lg-2">
        <span class="badge bg-primary d-flex align-items-center gap-1 px-3 py-2">
          <i class="bi bi-cpu"></i> IBM Granite
        </span>
      </li>
    </ul>
  </div>
</nav>

<!-- ═══════════════════════ HERO ═══════════════════════ -->
<div class="hero">
  <h1><i class="bi bi-shield-heart me-2"></i>MindGuard AI</h1>
  <p>
    An empathetic, AI-powered Early Distress Detection Assistant.
    Share what's on your mind — your journal entry, thoughts, or feelings —
    and receive compassionate, evidence-based emotional support.
  </p>
  <div class="badge-ibm">
    <i class="bi bi-cpu-fill"></i>
    Powered by IBM watsonx.ai · Granite Models · RAG Pipeline
  </div>
</div>

<!-- ═══════════════════════ MAIN CONTENT ═══════════════════════ -->
<div class="container py-4" style="max-width: 820px;">

  <!-- Disclaimer -->
  <div class="disclaimer">
    <i class="bi bi-info-circle-fill me-2"></i>
    <strong>Disclaimer:</strong> This AI assistant provides educational and emotional support only.
    It is <strong>not a substitute</strong> for professional medical advice, diagnosis, or treatment.
    If you are in crisis, please contact a mental health professional or emergency services immediately.
  </div>

  <!-- ── Tabs ── -->
  <ul class="nav nav-tabs mb-4" id="mainTabs" role="tablist">
    <li class="nav-item">
      <button class="nav-link active" id="tab-analyse" data-bs-toggle="tab"
              data-bs-target="#pane-analyse" role="tab">
        <i class="bi bi-journal-heart me-1"></i> Journal Analysis
      </button>
    </li>
    <li class="nav-item">
      <button class="nav-link" id="tab-resources" data-bs-toggle="tab"
              data-bs-target="#pane-resources" role="tab">
        <i class="bi bi-book-half me-1"></i> Mental Health Resources
      </button>
    </li>
    <li class="nav-item">
      <button class="nav-link" id="tab-about" data-bs-toggle="tab"
              data-bs-target="#pane-about" role="tab">
        <i class="bi bi-info-circle me-1"></i> About
      </button>
    </li>
  </ul>

  <div class="tab-content">

    <!-- ─────────────── ANALYSE TAB ─────────────── -->
    <div class="tab-pane fade show active" id="pane-analyse" role="tabpanel">

      <section id="analyseSection">
        <!-- Journal Input Card -->
        <div class="card journal-card mb-4">
          <div class="card-header py-3">
            <i class="bi bi-journal-text me-2"></i>Your Journal Entry
          </div>
          <div class="card-body p-4">

            <!-- Sample prompts -->
            <p class="text-muted small mb-2">
              <i class="bi bi-lightbulb me-1"></i>Try one of these prompts or write your own:
            </p>
            <div class="mb-3" id="promptChips">
              <span class="prompt-chip" data-text="I've been feeling anxious every day and I can't seem to relax.">😰 Daily Anxiety</span>
              <span class="prompt-chip" data-text="Nothing excites me anymore. I used to love my hobbies but now everything feels pointless.">😔 Loss of Interest</span>
              <span class="prompt-chip" data-text="I'm completely overwhelmed with work lately. I feel burned out and exhausted all the time.">🔥 Burnout</span>
              <span class="prompt-chip" data-text="I feel lonely even when I'm around people. I don't know how to connect with anyone.">🫂 Loneliness</span>
              <span class="prompt-chip" data-text="My exams are coming up and I can't stop stressing. I can't sleep and I feel sick with worry.">📚 Exam Stress</span>
            </div>

            <textarea
              id="journalText"
              class="form-control"
              placeholder="Write freely about how you've been feeling lately... your thoughts, emotions, challenges, or anything on your mind. This is a safe, private space."
              rows="7"
              maxlength="3000"
            ></textarea>
            <div class="char-count mt-1">
              <span id="charCount">0</span> / 3000 characters
            </div>

            <div class="d-flex align-items-center justify-content-between mt-4 flex-wrap gap-2">
              <button class="btn btn-outline-secondary btn-sm" id="clearBtn">
                <i class="bi bi-x-circle me-1"></i>Clear
              </button>
              <button class="btn btn-analyse px-4" id="analyseBtn">
                <i class="bi bi-brain me-2"></i>Analyse with MindGuard AI
              </button>
            </div>

          </div>
        </div>

        <!-- AI Loader -->
        <div class="ai-loader" id="aiLoader">
          <div class="loader-ring"></div>
          <div>
            <p class="fw-semibold mb-1">MindGuard AI is analysing your entry…</p>
            <div class="loader-steps" id="loaderStep">🔍 Retrieving relevant mental health resources…</div>
          </div>
        </div>

        <!-- Error Alert -->
        <div class="alert alert-danger d-none" id="errorAlert" role="alert">
          <i class="bi bi-exclamation-triangle-fill me-2"></i>
          <span id="errorMsg"></span>
        </div>

        <!-- ───── RESULTS ───── -->
        <section id="resultsSection">

          <!-- Crisis Banner -->
          <div class="crisis-banner d-none" id="crisisBanner">
            <h5><i class="bi bi-exclamation-octagon-fill me-2"></i>Immediate Support Available</h5>
            <p class="mb-2" id="crisisText"></p>
            <hr class="my-2" />
            <p class="mb-1 small fw-semibold">24/7 Crisis Resources:</p>
            <ul class="mb-0 small">
              <li><strong>988 Suicide &amp; Crisis Lifeline (US):</strong> Call or text <strong>988</strong></li>
              <li><strong>Crisis Text Line:</strong> Text HOME to <strong>741741</strong></li>
              <li><strong>International Crisis Centres:</strong>
                <a href="https://www.iasp.info/resources/Crisis_Centres/" target="_blank" rel="noopener">iasp.info</a>
              </li>
              <li><strong>Emergency Services:</strong> Call your local emergency number immediately</li>
            </ul>
          </div>

          <!-- Distress Level Card -->
          <div class="distress-card card mb-4 shadow-sm" id="distressCard">
            <div class="distress-header" id="distressHeader">
              <div>
                <div class="small text-white-50 mb-1">Overall Distress Level</div>
                <h3 class="mb-0 fw-bold" id="distressLevel">—</h3>
                <div class="small mt-1 opacity-75" id="emotionTagsInline"></div>
              </div>
              <div class="distress-score-circle">
                <span class="score-num" id="distressScoreNum">—</span>
                <span class="score-lbl">/ 10</span>
              </div>
            </div>
            <div class="card-body pt-3 pb-2">
              <p class="mb-0" id="emotionalSummary" style="font-size:.92rem;"></p>
            </div>
            <div class="meta-bar" id="metaBar">
              <span><i class="bi bi-cpu"></i><span id="metaModel">—</span></span>
              <span><i class="bi bi-clock"></i><span id="metaTime">—</span></span>
              <span><i class="bi bi-database"></i>RAG-Augmented Response</span>
            </div>
          </div>

          <!-- Affirmation -->
          <div class="affirmation-box" id="affirmationBox">
            <i class="bi bi-quote me-1"></i>
            <span id="affirmationText"></span>
          </div>

          <!-- Insights -->
          <div class="card result-card mb-3">
            <div class="card-header py-2 px-3">
              <i class="bi bi-lightbulb-fill"></i>Emotional Insights
            </div>
            <div class="card-body py-3 px-3" id="insightsList"></div>
          </div>

          <!-- Coping Strategies -->
          <div class="card result-card mb-3">
            <div class="card-header py-2 px-3">
              <i class="bi bi-heart-pulse-fill"></i>Personalised Coping Strategies
            </div>
            <div class="card-body py-3 px-3" id="copingList"></div>
          </div>

          <!-- Self-Care Activities -->
          <div class="card result-card mb-3">
            <div class="card-header py-2 px-3">
              <i class="bi bi-stars"></i>Self-Care Activities
            </div>
            <div class="card-body py-3 px-3" id="selfCareList"></div>
          </div>

          <!-- Professional Help -->
          <div class="prof-help-box d-none" id="profHelpBox">
            <h6><i class="bi bi-person-hearts me-2"></i>Consider Speaking with a Professional</h6>
            <p class="mb-1 small">
              Based on your entry, speaking with a licensed mental health professional could provide
              personalised support tailored to your needs. This is a sign of strength, not weakness.
            </p>
            <p class="mb-0 small">
              <strong>Find help:</strong>
              <a href="https://www.psychologytoday.com/us/therapists" target="_blank" rel="noopener">Psychology Today Therapist Finder</a> ·
              <a href="https://www.betterhelp.com" target="_blank" rel="noopener">BetterHelp</a> ·
              <a href="https://www.samhsa.gov/find-help/national-helpline" target="_blank" rel="noopener">SAMHSA Helpline</a>
            </p>
          </div>

          <!-- RAG Resources -->
          <div class="card result-card mb-3" id="ragResourceCard">
            <div class="card-header py-2 px-3">
              <i class="bi bi-database-check"></i>Retrieved Knowledge Resources
            </div>
            <div class="card-body py-3 px-3" id="ragResourcesList"></div>
          </div>

          <!-- New Analysis Button -->
          <div class="text-center mt-2 mb-4">
            <button class="btn btn-outline-primary" id="newAnalysisBtn">
              <i class="bi bi-arrow-counterclockwise me-2"></i>Start New Analysis
            </button>
          </div>

        </section>
      </section>
    </div><!-- /pane-analyse -->

    <!-- ─────────────── RESOURCES TAB ─────────────── -->
    <div class="tab-pane fade" id="pane-resources" role="tabpanel" id="resourcesSection">
      <h5 class="fw-bold mb-3"><i class="bi bi-book-half me-2 text-primary"></i>Mental Health Awareness Resources</h5>
      <p class="text-muted small mb-4">
        The following evidence-based resources are part of MindGuard AI's RAG knowledge base,
        retrieved automatically to ground AI responses in trusted mental health guidance.
      </p>
      <div id="resourceCards">
        <div class="text-center py-4 text-muted">
          <div class="spinner-border spinner-border-sm me-2"></div> Loading resources…
        </div>
      </div>
    </div>

    <!-- ─────────────── ABOUT TAB ─────────────── -->
    <div class="tab-pane fade" id="pane-about" role="tabpanel" id="aboutSection">
      <div class="card border-0 shadow-sm">
        <div class="card-body p-4">
          <h5 class="fw-bold mb-3"><i class="bi bi-shield-heart me-2 text-primary"></i>About MindGuard AI</h5>

          <h6 class="fw-semibold mt-3 mb-2">What is MindGuard AI?</h6>
          <p class="small text-muted">
            MindGuard AI is an Agentic AI-powered Early Distress Detection Assistant built on
            <strong>IBM watsonx.ai</strong> using <strong>IBM Granite language models</strong>.
            It analyses user-written journal entries to identify possible emotional distress indicators
            and provides empathetic, evidence-based support.
          </p>

          <h6 class="fw-semibold mt-3 mb-2">Technology Stack</h6>
          <div class="row g-2 mb-3">
            <div class="col-sm-6">
              <div class="d-flex align-items-center gap-2 p-2 rounded" style="background:#f0f4ff;">
                <i class="bi bi-cpu text-primary fs-5"></i>
                <div class="small"><strong>IBM watsonx.ai</strong><br/>Granite LLM inference</div>
              </div>
            </div>
            <div class="col-sm-6">
              <div class="d-flex align-items-center gap-2 p-2 rounded" style="background:#f0fff4;">
                <i class="bi bi-database text-success fs-5"></i>
                <div class="small"><strong>RAG Pipeline</strong><br/>Knowledge-grounded responses</div>
              </div>
            </div>
            <div class="col-sm-6">
              <div class="d-flex align-items-center gap-2 p-2 rounded" style="background:#fff8f0;">
                <i class="bi bi-braces text-warning fs-5"></i>
                <div class="small"><strong>Python / Flask</strong><br/>Backend API & routing</div>
              </div>
            </div>
            <div class="col-sm-6">
              <div class="d-flex align-items-center gap-2 p-2 rounded" style="background:#f9f0ff;">
                <i class="bi bi-layout-text-window text-purple fs-5" style="color:#8a3ffc"></i>
                <div class="small"><strong>Bootstrap 5</strong><br/>Responsive UI</div>
              </div>
            </div>
          </div>

          <h6 class="fw-semibold mt-3 mb-2">How the Agentic RAG Pipeline Works</h6>
          <ol class="small text-muted ps-3">
            <li class="mb-1"><strong>Input:</strong> User submits a journal entry or personal reflection.</li>
            <li class="mb-1"><strong>Retrieval:</strong> A keyword-overlap RAG engine retrieves the most relevant mental health knowledge base entries.</li>
            <li class="mb-1"><strong>Augmentation:</strong> Retrieved context is injected into a structured prompt alongside the journal entry.</li>
            <li class="mb-1"><strong>Generation:</strong> IBM Granite model analyses the augmented prompt and generates a structured JSON response.</li>
            <li class="mb-1"><strong>Output:</strong> The UI renders detected emotions, distress level, coping strategies, and relevant resources.</li>
          </ol>

          <h6 class="fw-semibold mt-3 mb-2">Setup &amp; Configuration</h6>
          <div class="small bg-dark text-light rounded p-3 font-monospace" style="font-size:.8rem;">
            <div class="text-success"># Set environment variables before running</div>
            export WATSONX_API_KEY="your-api-key"<br/>
            export WATSONX_URL="https://us-south.ml.cloud.ibm.com"<br/>
            export WATSONX_PROJECT_ID="your-project-id"<br/>
            export GRANITE_MODEL_ID="ibm/granite-13b-instruct-v2"<br/>
            <br/>
            <div class="text-success"># Install dependencies</div>
            pip install flask ibm-watsonx-ai<br/>
            <br/>
            <div class="text-success"># Run the application</div>
            python app.py
          </div>

          <div class="disclaimer mt-3">
            <i class="bi bi-info-circle-fill me-2"></i>
            <strong>Disclaimer:</strong> MindGuard AI provides educational and emotional support only.
            It is not a substitute for professional medical advice, diagnosis, or treatment.
            Always consult a qualified mental health professional for personalised care.
          </div>
        </div>
      </div>
    </div><!-- /pane-about -->

  </div><!-- /tab-content -->
</div><!-- /container -->

<!-- ═══════════════════════ FOOTER ═══════════════════════ -->
<footer>
  <div style="max-width:820px; margin:0 auto;">
    <p class="mb-1">
      <i class="bi bi-shield-heart me-1 text-primary"></i>
      <strong>MindGuard AI</strong> – Early Distress Detection Assistant
    </p>
    <p class="mb-2" style="font-size:.75rem;">
      Powered by <strong>IBM watsonx.ai</strong> · Granite Models · RAG Pipeline
    </p>
    <p class="mb-0" style="font-size:.72rem; color:#9ca3af;">
      This tool provides educational support only and is not a substitute for professional medical advice.
    </p>
  </div>
</footer>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

<script>
// ═══════════════════════════════════════════════════════════════
//  MindGuard AI – Front-End Controller
// ═══════════════════════════════════════════════════════════════

const $ = id => document.getElementById(id);

// ── Char counter ──
const journalText = $('journalText');
const charCount   = $('charCount');
journalText.addEventListener('input', () => {
  const len = journalText.value.length;
  charCount.textContent = len;
  charCount.parentElement.classList.toggle('warn', len > 2700);
});

// ── Sample prompt chips ──
document.querySelectorAll('.prompt-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    journalText.value = chip.dataset.text;
    journalText.dispatchEvent(new Event('input'));
    journalText.focus();
  });
});

// ── Clear button ──
$('clearBtn').addEventListener('click', () => {
  journalText.value = '';
  journalText.dispatchEvent(new Event('input'));
  hideResults();
});

// ── New Analysis button ──
$('newAnalysisBtn').addEventListener('click', () => {
  journalText.value = '';
  journalText.dispatchEvent(new Event('input'));
  hideResults();
  journalText.focus();
  journalText.scrollIntoView({ behavior: 'smooth', block: 'center' });
});

// ── Loader step messages ──
const loaderMessages = [
  '🔍 Retrieving relevant mental health resources…',
  '🧠 Analysing emotional patterns with IBM Granite…',
  '💡 Generating coping strategies and insights…',
  '📊 Classifying distress level…',
  '✅ Finalising your personalised report…',
];
let loaderInterval;
function startLoader() {
  let i = 0;
  $('loaderStep').textContent = loaderMessages[0];
  loaderInterval = setInterval(() => {
    i = (i + 1) % loaderMessages.length;
    $('loaderStep').textContent = loaderMessages[i];
  }, 1800);
}
function stopLoader() { clearInterval(loaderInterval); }

// ── Show / hide helpers ──
function showLoader() {
  $('aiLoader').classList.add('show');
  startLoader();
}
function hideLoader() {
  $('aiLoader').classList.remove('show');
  stopLoader();
}
function hideResults() {
  $('resultsSection').classList.remove('show');
  $('resultsSection').style.display = 'none';
  $('errorAlert').classList.add('d-none');
  $('crisisBanner').classList.add('d-none');
}
function showResults() {
  $('resultsSection').style.display = 'block';
  setTimeout(() => $('resultsSection').classList.add('show'), 10);
}

// ── Analyse button ──
$('analyseBtn').addEventListener('click', analyseEntry);

async function analyseEntry() {
  const text = journalText.value.trim();
  if (!text) { flashError('Please write something in the journal entry before analysing.'); return; }
  if (text.length < 10) { flashError('Your entry is too short. Please share a bit more.'); return; }

  hideResults();
  $('errorAlert').classList.add('d-none');
  $('analyseBtn').disabled = true;
  showLoader();

  try {
    const res = await fetch('/api/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ journal_text: text }),
    });
    const data = await res.json();

    if (!res.ok) {
      flashError(data.error || 'An unexpected error occurred. Please try again.');
      return;
    }

    renderResults(data);

  } catch (err) {
    flashError('Network error – please check your connection and try again.');
  } finally {
    hideLoader();
    $('analyseBtn').disabled = false;
  }
}

// ── Render Results ──
function renderResults(d) {
  // Crisis banner
  if (d.is_crisis && d.crisis_response) {
    $('crisisBanner').classList.remove('d-none');
    $('crisisText').textContent = d.crisis_response;
  } else {
    $('crisisBanner').classList.add('d-none');
  }

  // Distress level header
  const level  = (d.distress_level || 'Low').toLowerCase();
  const header = $('distressHeader');
  header.className = 'distress-header ' + level;
  $('distressLevel').textContent = d.distress_level || '—';
  $('distressScoreNum').textContent = d.distress_score || '—';

  // Emotion tags
  const emotions = d.detected_emotions || [];
  $('emotionTagsInline').innerHTML = emotions.map(e =>
    `<span class="emotion-tag" style="background:rgba(255,255,255,.2);color:#fff;">
      <i class="bi bi-circle-fill" style="font-size:.45rem;"></i>${e}
    </span>`
  ).join('');

  // Emotional summary
  $('emotionalSummary').textContent = d.emotional_summary || '';

  // Meta bar
  $('metaModel').textContent = d.model || 'IBM Granite';
  $('metaTime').textContent = d.timestamp || '';

  // Affirmation
  $('affirmationText').textContent = d.affirmation || '';

  // Insights
  const insights = d.insights || [];
  $('insightsList').innerHTML = insights.map(ins =>
    `<div class="insight-item">
      <div class="insight-dot"></div>
      <div>${escHtml(ins)}</div>
    </div>`
  ).join('') || '<p class="text-muted small mb-0">No insights generated.</p>';

  // Coping strategies
  const strategies = d.coping_strategies || [];
  $('copingList').innerHTML = strategies.map(s =>
    `<div class="strategy-card">
      <h6><i class="bi bi-check2-circle me-2"></i>${escHtml(s.title)}</h6>
      <p>${escHtml(s.description)}</p>
    </div>`
  ).join('') || '<p class="text-muted small mb-0">No strategies generated.</p>';

  // Self-care
  const selfCare = d.self_care_activities || [];
  $('selfCareList').innerHTML = selfCare.map(item =>
    `<div class="self-care-item">
      <i class="bi bi-flower1"></i>
      <span>${escHtml(item)}</span>
    </div>`
  ).join('') || '<p class="text-muted small mb-0">No activities generated.</p>';

  // Professional help box
  if (d.seek_professional_help) {
    $('profHelpBox').classList.remove('d-none');
  } else {
    $('profHelpBox').classList.add('d-none');
  }

  // RAG resources
  const rags = d.rag_resources || [];
  if (rags.length) {
    $('ragResourceCard').style.display = '';
    $('ragResourcesList').innerHTML = rags.map(r =>
      `<div class="rag-resource">
        <i class="bi bi-journal-bookmark-fill"></i>
        <div>
          <div class="small fw-semibold text-dark">${escHtml(r.topic)}</div>
          <a href="${escAttr(r.link)}" target="_blank" rel="noopener">${escHtml(r.resource)}</a>
        </div>
      </div>`
    ).join('');
  } else {
    $('ragResourceCard').style.display = 'none';
  }

  showResults();
  $('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Error helper ──
function flashError(msg) {
  $('errorMsg').textContent = msg;
  $('errorAlert').classList.remove('d-none');
  $('errorAlert').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ── Escape helpers ──
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function escAttr(str) {
  return String(str).replace(/"/g,'&quot;');
}

// ── Load resources tab ──
document.getElementById('tab-resources').addEventListener('click', loadResources);
async function loadResources() {
  const container = $('resourceCards');
  if (container.dataset.loaded) return;
  try {
    const res  = await fetch('/api/resources');
    const data = await res.json();
    container.innerHTML = data.map(r =>
      `<div class="resource-topic-card">
        <div class="rtc-header"><i class="bi bi-bookmark-fill me-2"></i>${escHtml(r.topic)}</div>
        <div class="rtc-body">
          <a href="${escAttr(r.link)}" target="_blank" rel="noopener" class="fw-semibold text-primary">
            <i class="bi bi-box-arrow-up-right me-1"></i>${escHtml(r.resource)}
          </a>
        </div>
      </div>`
    ).join('');
    container.dataset.loaded = '1';
  } catch {
    container.innerHTML = '<p class="text-danger small">Failed to load resources.</p>';
  }
}
</script>

</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print("=" * 60)
    print("  MindGuard AI - Early Distress Detection Assistant")
    print("  Powered by IBM watsonx.ai | Granite Models | RAG")
    print("=" * 60)
    print(f"  >> http://localhost:{port}")
    print(f"  >> Model : {GRANITE_MODEL_ID}")
    print(f"  >> Debug : {debug}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=debug)
