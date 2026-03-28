# ================================================================
#  Fake News Detector — Flask Web Application
#  Deploy on PythonAnywhere or Render (free tier)
#
#  Files required in the same directory as app.py:
#    fake_news_lr_model.pkl
#    tfidf_vectorizer.pkl
# ================================================================

from flask import Flask, request, render_template_string
import joblib, re

app = Flask(__name__)

# Load model and vectorizer once at server startup
model      = joblib.load("fake_news_lr_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Inline stopword list — no NLTK dependency needed on the server
STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','he','him','his','himself','she','her','hers','herself','it',
    'its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were',
    'be','been','being','have','has','had','having','do','does','did','doing',
    'a','an','the','and','but','if','or','because','as','until','while','of',
    'at','by','for','with','about','against','between','into','through',
    'during','before','after','above','below','to','from','up','down','in',
    'out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now'
}

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    text = ' '.join(w for w in text.split() if w not in STOP_WORDS)
    return text

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Fake News Detector</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
  <style>
    /* ── Reset & Base ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --rose:       #C75F71;
      --rose-dark:  #a84d5e;
      --blush:      #F0B8B8;
      --blush-light:#fdf0f0;
      --sage:       #A2AE9D;
      --sage-dark:  #8a9786;
      --brown:      #54463A;
      --brown-light:#6b5a4e;
      --white:      #ffffff;
      --shadow-sm:  0 2px 12px rgba(84,70,58,0.08);
      --shadow-md:  0 6px 28px rgba(84,70,58,0.13);
      --shadow-lg:  0 12px 40px rgba(84,70,58,0.18);
      --radius-sm:  10px;
      --radius-md:  14px;
      --radius-lg:  20px;
      --transition: 0.25s ease-in-out;
    }

    body {
      font-family: 'Poppins', sans-serif;
      background: linear-gradient(135deg, #fdf0f0 0%, #f5ebe0 50%, #eef1ee 100%);
      min-height: 100vh;
      color: var(--brown);
    }

    /* ── Navbar ── */
    nav {
      background: linear-gradient(90deg, var(--rose) 0%, #d4707f 60%, #c06b78 100%);
      padding: 0 32px;
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 3px 16px rgba(199,95,113,0.30);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .nav-brand {
      display: flex;
      align-items: center;
      gap: 10px;
      text-decoration: none;
    }
    .nav-icon {
      font-size: 22px;
    }
    .nav-title {
      font-size: 18px;
      font-weight: 600;
      color: var(--white);
      letter-spacing: 0.3px;
    }
    .nav-badge {
      background: rgba(255,255,255,0.20);
      color: var(--white);
      font-size: 11px;
      font-weight: 500;
      padding: 3px 10px;
      border-radius: 20px;
      border: 1px solid rgba(255,255,255,0.30);
    }

    /* ── Page wrapper ── */
    .page {
      max-width: 760px;
      margin: 0 auto;
      padding: 48px 24px 80px;
      animation: fadeSlideUp 0.5s ease-out both;
    }

    @keyframes fadeSlideUp {
      from { opacity: 0; transform: translateY(24px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Hero ── */
    .hero {
      text-align: center;
      margin-bottom: 36px;
    }
    .hero-emoji {
      font-size: 52px;
      display: block;
      margin-bottom: 12px;
      animation: pulse 3s ease-in-out infinite;
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); }
      50%       { transform: scale(1.06); }
    }
    .hero h1 {
      font-size: 30px;
      font-weight: 700;
      color: var(--brown);
      margin-bottom: 10px;
      line-height: 1.25;
    }
    .hero p {
      font-size: 15px;
      font-weight: 300;
      color: var(--brown-light);
      max-width: 500px;
      margin: 0 auto;
      line-height: 1.65;
    }

    /* ── Stats strip ── */
    .stats {
      display: flex;
      justify-content: center;
      gap: 16px;
      margin-bottom: 32px;
      flex-wrap: wrap;
    }
    .stat-pill {
      background: var(--white);
      border: 1px solid var(--blush);
      border-radius: 30px;
      padding: 7px 18px;
      font-size: 13px;
      font-weight: 500;
      color: var(--brown);
      box-shadow: var(--shadow-sm);
      display: flex;
      align-items: center;
      gap: 7px;
    }
    .stat-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--sage);
      display: inline-block;
    }

    /* ── Main card ── */
    .card {
      background: var(--white);
      border-radius: var(--radius-lg);
      padding: 36px 40px;
      box-shadow: var(--shadow-md);
      border: 1px solid rgba(240,184,184,0.35);
      margin-bottom: 24px;
      transition: box-shadow var(--transition);
    }
    .card:hover {
      box-shadow: var(--shadow-lg);
    }

    .card-label {
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      color: var(--sage-dark);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .card-label::before {
      content: '';
      display: inline-block;
      width: 18px; height: 2px;
      background: var(--sage);
      border-radius: 2px;
    }

    .card h2 {
      font-size: 19px;
      font-weight: 600;
      color: var(--brown);
      margin-bottom: 18px;
    }

    /* ── Textarea ── */
    textarea {
      width: 100%;
      min-height: 160px;
      padding: 18px 20px;
      font-family: 'Poppins', sans-serif;
      font-size: 14px;
      font-weight: 300;
      color: var(--brown);
      background: var(--blush-light);
      border: 2px solid var(--blush);
      border-radius: var(--radius-md);
      resize: vertical;
      outline: none;
      line-height: 1.7;
      transition: border-color var(--transition), box-shadow var(--transition), background var(--transition);
    }
    textarea::placeholder { color: #c4a0a0; }
    textarea:focus {
      border-color: var(--rose);
      background: var(--white);
      box-shadow: 0 0 0 4px rgba(199,95,113,0.10);
    }

    /* ── Char counter ── */
    .char-row {
      display: flex;
      justify-content: flex-end;
      margin-top: 8px;
    }
    .char-count {
      font-size: 12px;
      color: var(--sage-dark);
    }

    /* ── Button ── */
    .btn-primary {
      width: 100%;
      margin-top: 20px;
      padding: 15px 24px;
      background: linear-gradient(135deg, var(--rose) 0%, #d4707f 100%);
      color: var(--white);
      font-family: 'Poppins', sans-serif;
      font-size: 15px;
      font-weight: 600;
      border: none;
      border-radius: var(--radius-md);
      cursor: pointer;
      letter-spacing: 0.3px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      box-shadow: 0 4px 18px rgba(199,95,113,0.35);
      transition: transform var(--transition), box-shadow var(--transition), background var(--transition);
    }
    .btn-primary:hover {
      transform: translateY(-2px) scale(1.01);
      box-shadow: 0 8px 28px rgba(199,95,113,0.45);
      background: linear-gradient(135deg, var(--rose-dark) 0%, var(--rose) 100%);
    }
    .btn-primary:active {
      transform: translateY(0) scale(0.99);
    }

    /* ── Divider ── */
    .divider {
      display: flex;
      align-items: center;
      gap: 14px;
      margin: 28px 0;
    }
    .divider-line {
      flex: 1;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--blush), transparent);
    }
    .divider-text {
      font-size: 12px;
      color: var(--sage-dark);
      font-weight: 500;
      white-space: nowrap;
    }

    /* ── Result card ── */
    .result-card {
      border-radius: var(--radius-lg);
      padding: 32px 36px;
      text-align: center;
      animation: resultSlide 0.4s ease-out both;
      border: 2px solid;
    }
    @keyframes resultSlide {
      from { opacity: 0; transform: translateY(16px) scale(0.97); }
      to   { opacity: 1; transform: translateY(0)    scale(1);    }
    }

    .result-card.fake {
      background: linear-gradient(135deg, #fdf2f3 0%, #fde8ea 100%);
      border-color: var(--rose);
    }
    .result-card.real {
      background: linear-gradient(135deg, #f3f6f2 0%, #e8ede6 100%);
      border-color: var(--sage);
    }

    .result-icon { font-size: 44px; display: block; margin-bottom: 12px; }
    .result-label {
      font-size: 26px;
      font-weight: 700;
      margin-bottom: 8px;
      letter-spacing: 0.5px;
    }
    .fake  .result-label { color: var(--rose); }
    .real  .result-label { color: var(--sage-dark); }

    .result-sub {
      font-size: 14px;
      font-weight: 300;
      color: var(--brown-light);
      margin-bottom: 20px;
      line-height: 1.6;
    }

    /* ── Confidence bar ── */
    .conf-bar-wrap {
      background: rgba(255,255,255,0.7);
      border-radius: 30px;
      height: 10px;
      overflow: hidden;
      margin-bottom: 10px;
      box-shadow: inset 0 1px 3px rgba(0,0,0,0.06);
    }
    .conf-bar {
      height: 100%;
      border-radius: 30px;
      transition: width 1s ease;
    }
    .fake .conf-bar { background: linear-gradient(90deg, var(--blush), var(--rose)); }
    .real .conf-bar { background: linear-gradient(90deg, #c8d6c4, var(--sage-dark)); }
    .conf-text {
      font-size: 13px;
      font-weight: 500;
      color: var(--brown-light);
    }

    /* ── Input echo ── */
    .input-echo {
      margin-top: 20px;
      background: rgba(255,255,255,0.6);
      border-radius: var(--radius-sm);
      padding: 14px 18px;
      text-align: left;
      border: 1px solid rgba(0,0,0,0.05);
    }
    .input-echo-label {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1px;
      font-weight: 600;
      color: var(--sage-dark);
      margin-bottom: 6px;
    }
    .input-echo-text {
      font-size: 13px;
      font-weight: 300;
      color: var(--brown);
      line-height: 1.6;
      word-break: break-word;
    }

    /* ── Try again button ── */
    .btn-secondary {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-top: 20px;
      padding: 10px 24px;
      background: transparent;
      color: var(--rose);
      font-family: 'Poppins', sans-serif;
      font-size: 14px;
      font-weight: 500;
      border: 2px solid var(--rose);
      border-radius: var(--radius-sm);
      cursor: pointer;
      text-decoration: none;
      transition: background var(--transition), color var(--transition), transform var(--transition);
    }
    .btn-secondary:hover {
      background: var(--rose);
      color: var(--white);
      transform: translateY(-1px);
    }

    /* ── How it works ── */
    .how-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-top: 8px;
    }
    .how-item {
      background: var(--blush-light);
      border-radius: var(--radius-md);
      padding: 20px 16px;
      text-align: center;
      border: 1px solid var(--blush);
      transition: transform var(--transition), box-shadow var(--transition);
    }
    .how-item:hover {
      transform: translateY(-3px);
      box-shadow: var(--shadow-sm);
    }
    .how-num {
      width: 32px; height: 32px;
      background: var(--rose);
      color: var(--white);
      border-radius: 50%;
      font-size: 14px;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 10px;
    }
    .how-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--brown);
      margin-bottom: 5px;
    }
    .how-desc {
      font-size: 12px;
      font-weight: 300;
      color: var(--brown-light);
      line-height: 1.55;
    }

    /* ── Footer ── */
    footer {
      text-align: center;
      margin-top: 48px;
      padding-top: 24px;
      border-top: 1px solid var(--blush);
    }
    .footer-chips {
      display: flex;
      justify-content: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .chip {
      background: var(--white);
      border: 1px solid var(--blush);
      border-radius: 20px;
      padding: 5px 14px;
      font-size: 12px;
      font-weight: 500;
      color: var(--brown-light);
    }
    .footer-text {
      font-size: 12px;
      color: #b8a5a5;
    }

    /* ── Responsive ── */
    @media (max-width: 560px) {
      .card { padding: 24px 20px; }
      .hero h1 { font-size: 24px; }
      .how-grid { grid-template-columns: 1fr; }
      nav { padding: 0 16px; }
    }
  </style>
</head>
<body>

<!-- ── Navbar ── -->
<nav>
  <a class="nav-brand" href="/">
    <span class="nav-icon">📰</span>
    <span class="nav-title">Fake News Detector</span>
  </a>
  <span class="nav-badge">NLP · ML</span>
</nav>

<!-- ── Page ── -->
<div class="page">

  <!-- Hero -->
  <div class="hero">
    <span class="hero-emoji">🔍</span>
    <h1>Is this news real or fake?</h1>
    <p>Paste any news article — title and body text — and our AI will analyse it instantly using Natural Language Processing.</p>
  </div>

  <!-- Stats strip -->
  <div class="stats">
    <div class="stat-pill"><span class="stat-dot"></span> Logistic Regression</div>
    <div class="stat-pill"><span class="stat-dot"></span> TF-IDF Features</div>
    <div class="stat-pill"><span class="stat-dot"></span> ISOT Dataset</div>
  </div>

  {% if not prediction %}
  <!-- ── Input card ── -->
  <div class="card">
    <div class="card-label">Step 1</div>
    <h2>Paste your news article</h2>
    <form method="POST" action="/predict">
      <textarea
        id="newsInput"
        name="news_input"
        placeholder="Paste a news title and article body here…&#10;&#10;e.g. Trump to order withdrawal from Trans-Pacific Partnership&#10;&#10;WASHINGTON (Reuters) — President-elect Donald Trump plans to…"
        oninput="document.getElementById('charCount').textContent = this.value.length + ' characters'"
      >{{ news_input or '' }}</textarea>
      <div class="char-row">
        <span class="char-count" id="charCount">0 characters</span>
      </div>
      <button type="submit" class="btn-primary">
        <span>🧠</span> Analyse Article
      </button>
    </form>
  </div>

  <!-- How it works -->
  <div class="card">
    <div class="card-label">How it works</div>
    <div class="how-grid">
      <div class="how-item">
        <div class="how-num">1</div>
        <div class="how-title">You paste text</div>
        <div class="how-desc">Enter a news title and body text from any source.</div>
      </div>
      <div class="how-item">
        <div class="how-num">2</div>
        <div class="how-title">NLP processes it</div>
        <div class="how-desc">Text is cleaned and converted to TF-IDF feature vectors.</div>
      </div>
      <div class="how-item">
        <div class="how-num">3</div>
        <div class="how-title">AI predicts</div>
        <div class="how-desc">Logistic Regression classifies the article as REAL or FAKE.</div>
      </div>
    </div>
  </div>

  {% else %}
  <!-- ── Result card ── -->
  <div class="result-card {{ 'fake' if prediction == 'FAKE' else 'real' }}">
    <span class="result-icon">{{ '⚠️' if prediction == 'FAKE' else '✅' }}</span>
    <div class="result-label">
      {{ 'FAKE NEWS' if prediction == 'FAKE' else 'REAL NEWS' }}
    </div>
    <div class="result-sub">
      {% if prediction == 'FAKE' %}
        Our model predicts this article contains misleading or fabricated information.
      {% else %}
        Our model predicts this article is consistent with factual reporting.
      {% endif %}
    </div>

    <!-- Confidence bar -->
    <div class="conf-bar-wrap">
      <div class="conf-bar" style="width: {{ confidence }}%"></div>
    </div>
    <div class="conf-text">Model confidence: <strong>{{ confidence }}%</strong></div>

    <!-- Input echo -->
    <div class="input-echo">
      <div class="input-echo-label">Your input</div>
      <div class="input-echo-text">{{ news_input[:280] }}{% if news_input|length > 280 %}…{% endif %}</div>
    </div>

    <a href="/" class="btn-secondary">← Analyse another article</a>
  </div>

  <!-- How it works (still shown below result) -->
  <div class="card" style="margin-top:24px;">
    <div class="card-label">How it works</div>
    <div class="how-grid">
      <div class="how-item">
        <div class="how-num">1</div>
        <div class="how-title">You paste text</div>
        <div class="how-desc">Enter a news title and body text from any source.</div>
      </div>
      <div class="how-item">
        <div class="how-num">2</div>
        <div class="how-title">NLP processes it</div>
        <div class="how-desc">Text is cleaned and converted to TF-IDF feature vectors.</div>
      </div>
      <div class="how-item">
        <div class="how-num">3</div>
        <div class="how-title">AI predicts</div>
        <div class="how-desc">Logistic Regression classifies the article as REAL or FAKE.</div>
      </div>
    </div>
  </div>
  {% endif %}

  <!-- Footer -->
  <footer>
    <div class="footer-chips">
      <span class="chip">🤖 Logistic Regression</span>
      <span class="chip">📊 TF-IDF Vectorizer</span>
      <span class="chip">📰 ISOT Fake News Dataset</span>
      <span class="chip">🐍 Flask + Python</span>
    </div>
    <div class="footer-text">Built for educational purposes · NLP &amp; Machine Learning Assignment</div>
  </footer>

</div><!-- /page -->

<script>
  // Initialise char counter if text is pre-filled
  const ta = document.getElementById('newsInput');
  if (ta) {
    const cc = document.getElementById('charCount');
    cc.textContent = ta.value.length + ' characters';
  }
</script>
</body>
</html>"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    news_input = request.form.get("news_input", "").strip()
    cleaned    = preprocess(news_input)
    vec        = vectorizer.transform([cleaned])
    pred       = model.predict(vec)[0]
    proba      = model.predict_proba(vec)[0]
    confidence = round(max(proba) * 100, 1)
    return render_template_string(HTML,
                                  prediction=pred,
                                  confidence=confidence,
                                  news_input=news_input)

if __name__ == "__main__":
    app.run(debug=True)
