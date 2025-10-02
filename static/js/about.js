// about.js

// Translations - Updated with project & AI details
const translations = {
  en: {
    title: "About MGZon AI – Adaptive Horizons",
    heroTitle: "About MGZon AI",
    heroDesc: "An AI-powered platform revolutionizing e-commerce and code generation – Empowering creators and businesses with adaptive intelligence.",
    arHint: "Tap for AR View",
    storyTitle: "Our Journey",
    story1Title: "2023: Ignition in Alexandria",
    story1Desc: "MGZon AI sparks to life in Alexandria, Egypt, leveraging DeepSeek & FastAPI to launch the core chatbot – Empowering developers and e-commerce innovators globally.",
    story2Title: "2024: Global Expansion",
    story2Desc: "Seamless integrations with warehouses in USA, Canada, China; 5G/6G tech accelerates e-commerce automation and code generation for thousands.",
    story3Title: "2025-2026: Quantum Evolution",
    story3Desc: "Recognized by TechCrunch as MENA's AI leader; Pushing ethical AI boundaries in commerce and coding for a smarter future.",
    visionTitle: "Our Vision",
    visionDesc: "Adaptive Intelligence for Tomorrow",
    visionDetails: "MGZon AI fuses cutting-edge tech with real-world utility, democratizing AI for seamless e-commerce and effortless code creation.",
    visionBtn: "Explore Features",
    projectTitle: "Project & AI Insights",
    aiTechTitle: "Core AI Technologies",
    projectHowTitle: "How It Works",
    projectHowDesc: "AI analyzes user queries to generate code snippets or optimize e-commerce flows – Ethical, adaptive, and future-proof.",
    teamTitle: "Core Constellation",
    team1Name: "Sarah Dayan",
    team1Desc: "Lead AI Architect – Sculpting ML & chatbot symphonies for MGZon.",
    team2Name: "Leonardo Cuco",
    team2Desc: "E-Commerce Oracle – Orchestrating digital commerce evolutions for MGZon.",
    achievementsTitle: "Milestones",
    ach1Title: "Global Reach",
    ach1Desc: "Serving 10K+ users across 50 countries, powered by AI-driven constellations.",
    ach2Title: "Industry Acclaim",
    ach2Desc: "Featured as TechCrunch's top MENA AI innovator – Leading 2024's breakthroughs.",
    missionTitle: "Our North Star",
    missionDesc: "Democratizing AI for creators & enterprises – Streamlining codecraft and amplifying e-commerce realms with ethical innovation and ingenuity.",
    footerCrafted: "Powered by MGZon AI | Innovating in the MENA region",
    contact1Title: "Nexus Mail",
    contact1Details: "Transmit queries into the void – Responses warp in swiftly.",
    contact2Title: "Voice Link",
    contact2Details: "Instant quantum sync for urgent transmissions.",
    contact3Title: "Collective",
    contact3Btn: "Assemble",
    contact3Details: "Converge in the hive – Ideas ignite, collaborations supernova.",
    contact4Title: "Codex",
    contact4Btn: "Decrypt",
    contact4Details: "Unlock the arcane APIs – Seamless fusion awaits.",
    collapseBtn: "Collapse",
    copyright: "© 2026 MGZon AI. Eternal in the codeverse.",
    searchPlaceholder: "Search here... (e.g., AI trends 2026)",
    voiceTitle: "Press to speak",
    themeTitle: "Toggle Theme",
    chatPlaceholder: "Type your message...",
    sendBtn: "Send",
    voiceStatus: "Listening...",
    navLogo: "MGZon AI",
    navHome: "Home",
    navAbout: "About",
    navContact: "Contact",
    you: "You",
    error: "Hmm, I couldn't process that. Try asking about AI, team, or search something!",
    searchPrefix: "Here's what I found on the web:",
    searchError: "Sorry, search failed right now – try rephrasing or check your connection!",
    voicePrompt: "Voice enabled! Speak now."
  },
  ar: {
    title: "عن MGZon AI – آفاق متكيفة",
    heroTitle: "عن MGZon AI",
    heroDesc: "منصة مدعومة بالذكاء الاصطناعي ثورية في التجارة الإلكترونية وتوليد الكود – تمكين الخالقين والأعمال بالذكاء المتكيف.",
    arHint: "اضغط لعرض AR",
    storyTitle: "رحلتنا",
    story1Title: "2023: الإشعال في الإسكندرية",
    story1Desc: "MGZon AI يشتعل بالحياة في الإسكندرية، مصر، مستفيدًا من DeepSeek وFastAPI لإطلاق الشاتبوت الأساسي – تمكين المطورين ومبتكري التجارة الإلكترونية عالميًا.",
    story2Title: "2024: التوسع العالمي",
    story2Desc: "تكاملات سلسة مع مستودعات في الولايات المتحدة، كندا، الصين؛ تقنية 5G/6G تسرع أتمتة التجارة الإلكترونية وتوليد الكود لآلاف.",
    story3Title: "2025-2026: التطور الكمي",
    story3Desc: "معترف بها من TechCrunch كقائد AI في الشرق الأوسط؛ دفع حدود AI الأخلاقي في التجارة والكود لمستقبل أذكى.",
    visionTitle: "رؤيتنا",
    visionDesc: "ذكاء متكيف للغد",
    visionDetails: "MGZon AI يدمج التقنية المتقدمة مع المنفعة الواقعية، ديمقراطية AI لتجارة إلكترونية سلسة وإنشاء كود سهل.",
    visionBtn: "استكشف الميزات",
    projectTitle: "رؤى المشروع والذكاء الاصطناعي",
    aiTechTitle: "التقنيات الأساسية للذكاء الاصطناعي",
    projectHowTitle: "كيف يعمل",
    projectHowDesc: "الذكاء الاصطناعي يحلل استفسارات المستخدم لتوليد مقتطفات كود أو تحسين تدفقات التجارة الإلكترونية – أخلاقي، متكيف، ومستعد للمستقبل.",
    teamTitle: "الكوكبة الأساسية",
    team1Name: "سارة دايان",
    team1Desc: "مهندس AI رئيسي – نحت سيمفونيات ML وchatbot لـ MGZon.",
    team2Name: "ليوناردو كوكو",
    team2Desc: "خبير التجارة الإلكترونية – تنسيق تطورات التجارة الرقمية لـ MGZon.",
    achievementsTitle: "المعالم",
    ach1Title: "الوصول العالمي",
    ach1Desc: "خدمة 10K+ مستخدمين عبر 50 دولة، مدعومة بكوكبات مدفوعة بالذكاء الاصطناعي.",
    ach2Title: "الإعجاب الصناعي",
    ach2Desc: "مميزة كأفضل مبتكر AI في الشرق الأوسط من TechCrunch – قائدة اختراقات 2024.",
    missionTitle: "نجمنا الشمالي",
    missionDesc: "ديمقراطية AI للخالقين والمؤسسات – تبسيط صناعة الكود وتضخيم عوالم التجارة الإلكترونية بالابتكار الأخلاقي والإبداع.",
    footerCrafted: "مدعوم بـ MGZon AI | الابتكار في منطقة الشرق الأوسط",
    contact1Title: "بريد النيكسوس",
    contact1Details: "أرسل استفساراتك إلى الفراغ – الردود تتسارع بسرعة الضوء.",
    contact2Title: "رابط الصوت",
    contact2Details: "مزامنة كمية فورية للنقلات العاجلة.",
    contact3Title: "الجماعة",
    contact3Btn: "اجتمع",
    contact3Details: "اجتمع في النخلة – الأفكار تشعل، التعاونات سوبرنوفا.",
    contact4Title: "الكوديكس",
    contact4Btn: "فك الشفرة",
    contact4Details: "افتح APIs الغامضة – اندماج سلس ينتظر.",
    collapseBtn: "طي",
    copyright: "© 2026 MGZon AI. أبدي في عالم الكود.",
    searchPlaceholder: "ابحث هنا... (مثال: اتجاهات AI 2026)",
    voiceTitle: "اضغط للكلام",
    themeTitle: "تبديل الثيم",
    chatPlaceholder: "اكتب رسالتك...",
    sendBtn: "إرسال",
    voiceStatus: "يستمع...",
    navLogo: "MGZon AI",
    navHome: "الرئيسية",
    navAbout: "عننا",
    navContact: "اتصل",
    you: "أنت",
    error: "همم، مش قدرت أفهم ده. جرب تسأل عن AI أو الفريق أو ابحث في حاجة!",
    searchPrefix: "ده اللي لقيته على الويب:",
    searchError: "آسف، البحث فشل دلوقتي – جرب تعيد الصياغة أو تأكد من الإنترنت!",
    voicePrompt: "الصوت مفعل! اتكلم دلوقتي."
  },
  fr: {
    title: "À propos de MGZon AI – Horizons Adaptatifs",
    heroTitle: "À propos de MGZon AI",
    heroDesc: "Plateforme alimentée par l'IA révolutionnant le e-commerce et la génération de code – Empowerant les créateurs et entreprises avec une intelligence adaptative.",
    arHint: "Appuyez pour la vue AR",
    storyTitle: "Notre Voyage",
    story1Title: "2023: Allumage à Alexandrie",
    story1Desc: "MGZon AI s'allume à Alexandrie, Égypte, exploitant DeepSeek & FastAPI pour lancer le chatbot central – Empowerant les développeurs et innovateurs e-commerce globalement.",
    story2Title: "2024: Expansion Globale",
    story2Desc: "Intégrations fluides avec des entrepôts aux USA, Canada, Chine ; La tech 5G/6G accélère l'automatisation e-commerce et la génération de code pour des milliers.",
    story3Title: "2025-2026: Évolution Quantique",
    story3Desc: "Reconnue par TechCrunch comme leader AI MENA ; Poussant les frontières éthiques de l'AI en commerce et codage pour un avenir plus intelligent.",
    visionTitle: "Notre Vision",
    visionDesc: "Intelligence Adaptative pour Demain",
    visionDetails: "MGZon AI fusionne tech de pointe avec utilité réelle, démocratisant l'AI pour e-commerce fluide et création de code sans effort.",
    visionBtn: "Explorer les Fonctionnalités",
    projectTitle: "Perspectives du Projet & IA",
    aiTechTitle: "Technologies IA de Base",
    projectHowTitle: "Comment Ça Marche",
    projectHowDesc: "L'IA analyse les requêtes des utilisateurs pour générer des extraits de code ou optimiser les flux e-commerce – Éthique, adaptatif et prêt pour l'avenir.",
    teamTitle: "Constellation Centrale",
    team1Name: "Sarah Dayan",
    team1Desc: "Architecte AI Principal – Sculptant symphonies ML & chatbot pour MGZon.",
    team2Name: "Leonardo Cuco",
    team2Desc: "Oracle E-Commerce – Orchestrant évolutions commerce numérique pour MGZon.",
    achievementsTitle: "Jalons",
    ach1Title: "Portée Globale",
    ach1Desc: "Servant 10K+ utilisateurs dans 50 pays, alimenté par constellations AI.",
    ach2Title: "Acclamation Industrielle",
    ach2Desc: "Mise en avant comme top innovateur AI MENA par TechCrunch – Menant les percées 2024.",
    missionTitle: "Notre Étoile Polaire",
    missionDesc: "Démocratiser l'AI pour créateurs & entreprises – Rationalisant l'art du code et amplifiant les royaumes e-commerce avec innovation éthique et ingéniosité.",
    footerCrafted: "Alimenté par MGZon AI | Innovant dans la région MENA",
    contact1Title: "Courrier Nexus",
    contact1Details: "Transmettez les requêtes dans le vide – Les réponses se déforment rapidement.",
    contact2Title: "Lien Vocal",
    contact2Details: "Synchronisation quantique instantanée pour transmissions urgentes.",
    contact3Title: "Collectif",
    contact3Btn: "Assembler",
    contact3Details: "Converger dans la ruche – Idées s'allument, collaborations supernova.",
    contact4Title: "Codex",
    contact4Btn: "Décrypter",
    contact4Details: "Déverrouillez les APIs arcanes – Fusion seamless attend.",
    collapseBtn: "Réduire",
    copyright: "© 2026 MGZon AI. Éternel dans le codeverse.",
    searchPlaceholder: "Rechercher ici... (ex: tendances AI 2026)",
    voiceTitle: "Appuyez pour parler",
    themeTitle: "Changer le thème",
    chatPlaceholder: "Tapez votre message...",
    sendBtn: "Envoyer",
    voiceStatus: "Écoute...",
    navLogo: "MGZon AI",
    navHome: "Accueil",
    navAbout: "À propos",
    navContact: "Contact",
    you: "Vous",
    error: "Hmm, je n'ai pas pu traiter cela. Essayez de demander sur l'AI, l'équipe ou recherchez quelque chose !",
    searchPrefix: "Voici ce que j'ai trouvé sur le web:",
    searchError: "Désolé, la recherche a échoué pour le moment – essayez de reformuler ou vérifiez votre connexion !",
    voicePrompt: "Voix activée ! Parlez maintenant."
  },
  de: {
    title: "Über MGZon AI – Adaptive Horizonte",
    heroTitle: "Über MGZon AI",
    heroDesc: "AI-gestützte Plattform, die E-Commerce und Code-Generierung revolutioniert – Ermächtigt Schöpfer und Unternehmen mit adaptiver Intelligenz.",
    arHint: "Tippen Sie für AR-Ansicht",
    storyTitle: "Unsere Reise",
    story1Title: "2023: Zündung in Alexandria",
    story1Desc: "MGZon AI entzündet sich in Alexandria, Ägypten, nutzt DeepSeek & FastAPI, um den Kern-Chatbot zu starten – Ermächtigt Entwickler und E-Commerce-Innovatoren global.",
    story2Title: "2024: Globale Expansion",
    story2Desc: "Nahtlose Integrationen mit Lagern in USA, Kanada, China; 5G/6G-Tech beschleunigt E-Commerce-Automatisierung und Code-Generierung für Tausende.",
    story3Title: "2025-2026: Quantenevolution",
    story3Desc: "Von TechCrunch als MENA AI-Führer anerkannt; Drängt ethische AI-Grenzen in Handel und Codierung für eine intelligentere Zukunft.",
    visionTitle: "Unsere Vision",
    visionDesc: "Adaptive Intelligenz für Morgen",
    visionDetails: "MGZon AI fusioniert Spitzen-Tech mit realer Nützlichkeit, demokratisiert AI für nahtloses E-Commerce und mühelose Code-Erstellung.",
    visionBtn: "Funktionen Erkunden",
    projectTitle: "Projekt- & KI-Einblicke",
    aiTechTitle: "Kern-KI-Technologien",
    projectHowTitle: "Wie Es Funktioniert",
    projectHowDesc: "KI analysiert Nutzeranfragen, um Codeschnipsel zu generieren oder E-Commerce-Flüsse zu optimieren – Ethisch, adaptiv und zukunftssicher.",
    teamTitle: "Kernkonstellation",
    team1Name: "Sarah Dayan",
    team1Desc: "Lead AI-Architekt – Modelliert ML & Chatbot-Symphonien für MGZon.",
    team2Name: "Leonardo Cuco",
    team2Desc: "E-Commerce-Oracle – Dirigiert digitale Commerce-Evolutionen für MGZon.",
    achievementsTitle: "Meilensteine",
    ach1Title: "Globale Reichweite",
    ach1Desc: "Bedient 10K+ Nutzer in 50 Ländern, angetrieben von AI-Konstellationen.",
    ach2Title: "Branchen-Anerkennung",
    ach2Desc: "Vorgestellt als Top MENA AI-Innovator von TechCrunch – Führend 2024s Durchbrüche.",
    missionTitle: "Unser Nordstern",
    missionDesc: "Demokratisierung von AI für Schöpfer & Unternehmen – Vereinfacht Codekunst und verstärkt E-Commerce-Reiche mit ethischer Innovation und Erfindungsgabe.",
    footerCrafted: "Angetrieben von MGZon AI | Innovierend in der MENA-Region",
    contact1Title: "Nexus-Mail",
    contact1Details: "Übermitteln Sie Anfragen ins Leere – Antworten verbiegen sich schnell.",
    contact2Title: "Sprachlink",
    contact2Details: "Sofortige Quantensynchronisation für dringende Übertragungen.",
    contact3Title: "Kollektiv",
    contact3Btn: "Versammeln",
    contact3Details: "Konvergieren im Stock – Ideen entzünden, Kollaborationen Supernova.",
    contact4Title: "Codex",
    contact4Btn: "Entschlüsseln",
    contact4Details: "Entsperren Sie die arkane APIs – Nahtlose Fusion wartet.",
    collapseBtn: "Zusammenklappen",
    copyright: "© 2026 MGZon AI. Ewig im Codeversum.",
    searchPlaceholder: "Suchen Sie hier... (z.B. AI-Trends 2026)",
    voiceTitle: "Drücken zum Sprechen",
    themeTitle: "Thema wechseln",
    chatPlaceholder: "Geben Sie Ihre Nachricht ein...",
    sendBtn: "Senden",
    voiceStatus: "Hört zu...",
    navLogo: "MGZon AI",
    navHome: "Home",
    navAbout: "Über uns",
    navContact: "Kontakt",
    you: "Sie",
    error: "Hmm, ich konnte das nicht verarbeiten. Versuchen Sie, nach AI, Team oder etwas zu suchen!",
    searchPrefix: "Hier ist, was ich im Web gefunden habe:",
    searchError: "Entschuldigung, die Suche ist gerade fehlgeschlagen – versuchen Sie, umzuformulieren oder überprüfen Sie Ihre Verbindung!",
    voicePrompt: "Sprache aktiviert! Sprechen Sie jetzt."
  }
};

// Language and Theme Management
let currentLang = navigator.language.split('-')[0] || 'en';
if (!translations[currentLang]) currentLang = 'en';

function switchLanguage(lang) {
  currentLang = lang;
  document.querySelectorAll('[data-translate]').forEach(el => {
    const key = el.dataset.translate;
    if (translations[lang][key]) {
      el.textContent = translations[lang][key];
    }
  });
  document.querySelectorAll('[data-translate-placeholder]').forEach(el => {
    const key = el.dataset.translatePlaceholder;
    if (translations[lang][key]) {
      el.placeholder = translations[lang][key];
    }
  });
  document.querySelectorAll('[data-title-translate]').forEach(el => {
    const key = el.dataset.titleTranslate;
    if (translations[lang][key]) {
      el.title = translations[lang][key];
    }
  });
  const titleKey = document.title.dataset.translate || 'title';
  if (translations[lang][titleKey]) {
    document.title = translations[lang][titleKey];
  }
  document.querySelector('meta[property="og:title"]').setAttribute('content', translations[lang].title || document.title);
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
  document.body.classList.toggle('rtl', lang === 'ar');
  if (recognition) recognition.lang = lang === 'ar' ? 'ar-SA' : lang === 'fr' ? 'fr-FR' : lang === 'de' ? 'de-DE' : 'en-US';
}

// Theme Toggle
document.getElementById('theme-toggle').addEventListener('click', () => {
  document.body.dataset.theme = document.body.dataset.theme === 'light' ? 'dark' : 'light';
  const icon = document.querySelector('#theme-toggle i');
  icon.classList.toggle('bx-sun');
  icon.classList.toggle('bx-moon');
});

// Language Toggle
document.getElementById('lang-toggle').addEventListener('change', (e) => {
  switchLanguage(e.target.value);
});

// GSAP Animations
gsap.registerPlugin(ScrollTrigger);
gsap.from(".timeline-node", {
  scrollTrigger: { trigger: ".timeline-container", start: "top 80%" },
  y: 60,
  opacity: 0,
  duration: 1.2,
  stagger: 0.3,
  ease: "power3.out"
});
gsap.from(".neon-gradient", {
  scrollTrigger: { trigger: "section" },
  scale: 0.8,
  opacity: 0,
  duration: 1,
  ease: "back.out(1.7)"
});

// Three.js Orb
const orbScene = new THREE.Scene();
const orbCamera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
const orbRenderer = new THREE.WebGLRenderer({ canvas: document.getElementById('hero-orb') || document.createElement('canvas'), alpha: true });
orbRenderer.setSize(300, 300);
const orbGeometry = new THREE.SphereGeometry(1, 32, 32);
const orbMaterial = new THREE.MeshBasicMaterial({ color: 0x00f0ff, wireframe: true });
const orb = new THREE.Mesh(orbGeometry, orbMaterial);
orbScene.add(orb);
orbCamera.position.z = 5;
function rotateOrb() {
  requestAnimationFrame(rotateOrb);
  orb.rotation.x += 0.005;
  orb.rotation.y += 0.01;
  orbRenderer.render(orbScene, orbCamera);
}
rotateOrb();

// Particles.js (Optimized for performance)
particlesJS('particles-js', {
  particles: {
    number: { value: 20, density: { enable: true, value_area: 800 } },
    color: { value: ['#00f0ff', '#ff007a', '#6b21a8'] },
    shape: { type: 'star' },
    opacity: { value: 0.3, random: true },
    size: { value: 2, random: true },
    line_linked: { enable: false },
    move: { enable: true, speed: 2, direction: 'none', random: true, out_mode: 'out' }
  },
  interactivity: { events: { onhover: { enable: true, mode: 'bubble' } } }
});

// 3D Ambient (Optimized)
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('3d-canvas'), alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const geometry = new THREE.BoxGeometry();
const material = new THREE.MeshBasicMaterial({ color: 0x00f0ff, wireframe: true });
for (let i = 0; i < 15; i++) {
  const cube = new THREE.Mesh(geometry, material);
  cube.position.set((Math.random() - 0.6) * 100, (Math.random() - 0.6) * 100, (Math.random() - 0.6) * 100);
  scene.add(cube);
}
camera.position.z = 50;
function animate3D() {
  requestAnimationFrame(animate3D);
  scene.children.forEach(child => {
    child.rotation.x += 0.005;
    child.rotation.y += 0.01;
  });
  renderer.render(scene, camera);
}
animate3D();

// AR Toggle
document.querySelector('.ar-hint').addEventListener('click', () => {
  const arScene = document.getElementById('ar-scene');
  arScene.style.display = arScene.style.display === 'none' ? 'block' : 'none';
});

// Chat and Search
let conversationHistory = [];

// TTS function
function speakText(text) {
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = currentLang === 'ar' ? 'ar-SA' : currentLang === 'fr' ? 'fr-FR' : currentLang === 'de' ? 'de-DE' : 'en-US';
    utterance.rate = 0.9;
    utterance.pitch = 1;
    speechSynthesis.speak(utterance);
  } else {
    console.log('TTS not supported');
  }
}

// Enhanced search function (محدث للاتصال بالخادم الخلفي /search)
async function searxSearch(query) {
  const lowerQuery = query.toLowerCase();
  if (lowerQuery.includes('hello') || lowerQuery.includes('hi') || lowerQuery.includes('how are you') || lowerQuery.includes('ازيك') || lowerQuery.includes('مرحبا')) {
    query = `funny or friendly response to "${query}"`;
  }

  try {
    // الاتصال بالخادم الخلفي (بدون CORS مشكلة لأنه من نفس الدومين)
    const backendUrl = '/search';  // أو 'https://mgzon-mgzon-app.hf.space/search' لو على HF
    const response = await fetch(`${backendUrl}?q=${encodeURIComponent(query)}`, {
      headers: { 'Accept': 'application/json' }
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.success && data.results && data.results.length > 0) {
      const resultsText = data.results.map((result, i) => 
        `Result ${i + 1}:\nTitle: ${result.title}\nLink: ${result.link}\nContent: ${result.content.substring(0, 200)}...`
      ).join('\n\n');
      return { success: true, text: translations[currentLang].searchPrefix + '\n\n' + resultsText };
    }
    return { success: false, text: translations[currentLang].searchError };
  } catch (err) {
    console.log(`Search failed: ${err.message}`);
    return { success: false, text: translations[currentLang].searchError };
  }
}

// Bot response function
async function getBotResponse(userText) {
  const searchResult = await searxSearch(userText);
  const reply = searchResult.success ? searchResult.text : searchResult.text;
  speakText(reply);
  return reply;
}

// Enhanced chat display
function updateChatDisplay(userText, botText) {
  const userDiv = document.createElement('div');
  userDiv.className = 'chat-message user';
  userDiv.innerHTML = `<strong>${translations[currentLang].you}:</strong> ${userText}`;
  
  const botDiv = document.createElement('div');
  botDiv.className = 'chat-message';
  const formattedText = botText.split('\n\n').map(line => `<p>${line.replace(/\n/g, '<br>')}</p>`).join('');
  botDiv.innerHTML = `<strong>MGZon AI:</strong><div>${formattedText}</div>`;
  
  const popupContent = document.getElementById('popup-content');
  popupContent.appendChild(userDiv);
  popupContent.appendChild(botDiv);
  const popup = document.getElementById('gemini-popup');
  popup.style.display = 'block';
  popupContent.scrollTop = popupContent.scrollHeight;
  gsap.from(popup, { scale: 0.8, opacity: 0, duration: 0.5 });
}

// Search Input
document.getElementById('search-input').addEventListener('keypress', async (e) => {
  if (e.key === 'Enter') {
    const query = e.target.value;
    if (query) {
      conversationHistory.push({ role: 'user', text: query });
      const reply = await getBotResponse(query);
      conversationHistory.push({ role: 'assistant', text: reply });
      updateChatDisplay(query, reply);
      e.target.value = '';
    }
  }
});

// Chat Send
document.getElementById('send-chat').addEventListener('click', async () => {
  const query = document.getElementById('chat-input').value;
  if (query) {
    conversationHistory.push({ role: 'user', text: query });
    const reply = await getBotResponse(query);
    conversationHistory.push({ role: 'assistant', text: reply });
    updateChatDisplay(query, reply);
    document.getElementById('chat-input').value = '';
  }
});

document.getElementById('chat-input').addEventListener('keypress', (e) => {
  if (e.key === 'Enter') document.getElementById('send-chat').click();
});

function closePopup() {
  document.getElementById('gemini-popup').style.display = 'none';
}

function togglePanel(id) {
  const panel = document.getElementById(id + '-panel');
  panel.style.display = panel.style.display === 'block' ? 'none' : 'block';
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Enhanced Voice Recognition (Non-continuous, with repeat filter)
const voiceBtn = document.getElementById('voice-toggle');
const statusDiv = document.getElementById('voice-status');
let recognition;
let isListening = false;
let lastTranscript = '';
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.lang = currentLang === 'ar' ? 'ar-SA' : currentLang === 'fr' ? 'fr-FR' : currentLang === 'de' ? 'de-DE' : 'en-US';
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onstart = () => {
    isListening = true;
    voiceBtn.classList.add('listening');
    statusDiv.style.display = 'block';
    statusDiv.textContent = translations[currentLang].voiceStatus || 'Listening...';
  };

  recognition.onresult = async (event) => {
    const transcript = event.results[0][0].transcript;
    if (transcript === lastTranscript) return; // Prevent duplicates
    lastTranscript = transcript;
    statusDiv.textContent = `Heard: ${transcript}`;
    conversationHistory.push({ role: 'user', text: transcript });
    const reply = await getBotResponse(transcript);
    conversationHistory.push({ role: 'assistant', text: reply });
    updateChatDisplay(transcript, reply);
  };

  recognition.onerror = (event) => {
    statusDiv.textContent = `Error: ${event.error}. Try again!`;
    isListening = false;
    voiceBtn.classList.remove('listening');
    setTimeout(() => statusDiv.style.display = 'none', 3000);
  };

  recognition.onend = () => {
    isListening = false;
    voiceBtn.classList.remove('listening');
    setTimeout(() => statusDiv.style.display = 'none', 2000);
  };

  voiceBtn.addEventListener('click', () => {
    if (speechSynthesis.speaking) speechSynthesis.cancel();
    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
    }
  });

  voiceBtn.title = translations[currentLang].voiceTitle || 'Press to speak';
} else {
  voiceBtn.disabled = true;
  voiceBtn.title = 'Voice not supported. Use Chrome!';
  statusDiv.style.display = 'block';
  statusDiv.textContent = 'Voice not supported – Use Chrome!';
}

// Initialize
switchLanguage(currentLang);
