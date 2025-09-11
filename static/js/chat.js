// SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
// SPDX-License-Identifier: Apache-2.0

// Prism for code highlighting
Prism.plugins.autoloader.languages_path = 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/';

// UI elements
const uiElements = {
  chatArea: document.getElementById('chatArea'),
  chatBox: document.getElementById('chatBox'),
  initialContent: document.getElementById('initialContent'),
  form: document.getElementById('footerForm'),
  input: document.getElementById('userInput'),
  sendBtn: document.getElementById('sendBtn'),
  stopBtn: document.getElementById('stopBtn'),
  fileBtn: document.getElementById('fileBtn'),
  audioBtn: document.getElementById('audioBtn'),
  fileInput: document.getElementById('fileInput'),
  audioInput: document.getElementById('audioInput'),
  filePreview: document.getElementById('filePreview'),
  audioPreview: document.getElementById('audioPreview'),
  promptItems: document.querySelectorAll('.prompt-item'),
  chatHeader: document.getElementById('chatHeader'),
  clearBtn: document.getElementById('clearBtn'),
  messageLimitWarning: document.getElementById('messageLimitWarning'),
  conversationTitle: document.getElementById('conversationTitle'),
  sidebar: document.getElementById('sidebar'),
  sidebarToggle: document.getElementById('sidebarToggle'),
  conversationList: document.getElementById('conversationList'),
  newConversationBtn: document.getElementById('newConversationBtn'),
  swipeHint: document.getElementById('swipeHint'),
  settingsBtn: document.getElementById('settingsBtn'),
  settingsModal: document.getElementById('settingsModal'),
  closeSettingsBtn: document.getElementById('closeSettingsBtn'),
  settingsForm: document.getElementById('settingsForm'),
  historyToggle: document.getElementById('historyToggle')
};

// Track state
let streamMsg = null;
let conversationHistory = JSON.parse(sessionStorage.getItem('conversationHistory') || '[]');
let currentAssistantText = '';
let isRequestActive = false;
let abortController = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let currentConversationId = window.conversationId || null;
let currentConversationTitle = window.conversationTitle || null;
let isSidebarOpen = false;

// Auto-resize textarea
function autoResizeTextarea() {
  if (uiElements.input) {
    uiElements.input.style.height = 'auto';
    uiElements.input.style.height = `${Math.min(uiElements.input.scrollHeight, 200)}px`;
  }
}

// Detect Arabic text
function isArabicText(text) {
  return /[\u0600-\u06FF]/.test(text);
}

// Initialize page
document.addEventListener('DOMContentLoaded', async () => {
  AOS.init({
    duration: 800,
    easing: 'ease-out-cubic',
    once: true,
    offset: 50,
  });
  if (currentConversationId && checkAuth()) {
    await loadConversation(currentConversationId);
  } else if (conversationHistory.length > 0) {
    enterChatView();
    conversationHistory.forEach(msg => addMsg(msg.role, msg.content));
  }
  if (checkAuth()) {
    await loadConversations();
  }
  autoResizeTextarea();
  updateSendButtonState();
  if (uiElements.swipeHint) {
    setTimeout(() => {
      uiElements.swipeHint.style.display = 'none';
    }, 3000);
  }
  setupTouchGestures();
});

// Check authentication token
function checkAuth() {
  return localStorage.getItem('token');
}

// Update send button state
function updateSendButtonState() {
  if (uiElements.sendBtn && uiElements.input && uiElements.fileInput && uiElements.audioInput) {
    uiElements.sendBtn.disabled = uiElements.input.value.trim() === '' &&
      uiElements.fileInput.files.length === 0 &&
      uiElements.audioInput.files.length === 0;
  }
}

// Render markdown content with RTL support
function renderMarkdown(el) {
  const raw = el.dataset.text || '';
  const isArabic = isArabicText(raw);
  const html = marked.parse(raw, {
    gfm: true,
    breaks: true,
    smartLists: true,
    smartypants: false,
    headerIds: false,
  });
  el.innerHTML = `<div class="md-content" style="direction: ${isArabic ? 'rtl' : 'ltr'}; text-align: ${isArabic ? 'right' : 'left'};">${html}</div>`;
  const wrapper = el.querySelector('.md-content');
  wrapper.querySelectorAll('table').forEach(t => {
    if (!t.parentNode.classList?.contains('table-wrapper')) {
      const div = document.createElement('div');
      div.className = 'table-wrapper';
      t.parentNode.insertBefore(div, t);
      div.appendChild(t);
    }
  });
  wrapper.querySelectorAll('hr').forEach(h => h.classList.add('styled-hr'));
  Prism.highlightAllUnder(wrapper);
}

// Toggle chat view
function enterChatView() {
  if (uiElements.chatHeader) {
    uiElements.chatHeader.classList.remove('hidden');
    uiElements.chatHeader.setAttribute('aria-hidden', 'false');
    if (currentConversationTitle && uiElements.conversationTitle) {
      uiElements.conversationTitle.textContent = currentConversationTitle;
    }
  }
  if (uiElements.chatBox) uiElements.chatBox.classList.remove('hidden');
  if (uiElements.initialContent) uiElements.initialContent.classList.add('hidden');
}

// Toggle home view
function leaveChatView() {
  if (uiElements.chatHeader) {
    uiElements.chatHeader.classList.add('hidden');
    uiElements.chatHeader.setAttribute('aria-hidden', 'true');
  }
  if (uiElements.chatBox) uiElements.chatBox.classList.add('hidden');
  if (uiElements.initialContent) uiElements.initialContent.classList.remove('hidden');
}

// Add chat bubble
function addMsg(who, text) {
  const div = document.createElement('div');
  div.className = `bubble ${who === 'user' ? 'bubble-user' : 'bubble-assist'} ${isArabicText(text) ? 'rtl' : ''}`;
  div.dataset.text = text;
  renderMarkdown(div);
  if (uiElements.chatBox) {
    uiElements.chatBox.appendChild(div);
    uiElements.chatBox.classList.remove('hidden');
    uiElements.chatBox.scrollTop = uiElements.chatBox.scrollHeight;
  }
  return div;
}

// Clear all messages
function clearAllMessages() {
  stopStream(true);
  conversationHistory = [];
  sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
  currentAssistantText = '';
  if (streamMsg) {
    streamMsg.querySelector('.loading')?.remove();
    streamMsg = null;
  }
  if (uiElements.chatBox) uiElements.chatBox.innerHTML = '';
  if (uiElements.input) uiElements.input.value = '';
  if (uiElements.sendBtn) uiElements.sendBtn.disabled = true;
  if (uiElements.stopBtn) uiElements.stopBtn.style.display = 'none';
  if (uiElements.sendBtn) uiElements.sendBtn.style.display = 'inline-flex';
  if (uiElements.filePreview) uiElements.filePreview.style.display = 'none';
  if (uiElements.audioPreview) uiElements.audioPreview.style.display = 'none';
  if (uiElements.messageLimitWarning) uiElements.messageLimitWarning.classList.add('hidden');
  currentConversationId = null;
  currentConversationTitle = null;
  if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = 'MGZon AI Assistant';
  enterChatView();
  autoResizeTextarea();
}

// File preview
function previewFile() {
  if (uiElements.fileInput?.files.length > 0) {
    const file = uiElements.fileInput.files[0];
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = e => {
        if (uiElements.filePreview) {
          uiElements.filePreview.innerHTML = `<img src="${e.target.result}" class="upload-preview">`;
          uiElements.filePreview.style.display = 'block';
        }
        if (uiElements.audioPreview) uiElements.audioPreview.style.display = 'none';
        updateSendButtonState();
      };
      reader.readAsDataURL(file);
    }
  }
  if (uiElements.audioInput?.files.length > 0) {
    const file = uiElements.audioInput.files[0];
    if (file.type.startsWith('audio/')) {
      const reader = new FileReader();
      reader.onload = e => {
        if (uiElements.audioPreview) {
          uiElements.audioPreview.innerHTML = `<audio controls src="${e.target.result}"></audio>`;
          uiElements.audioPreview.style.display = 'block';
        }
        if (uiElements.filePreview) uiElements.filePreview.style.display = 'none';
        updateSendButtonState();
      };
      reader.readAsDataURL(file);
    }
  }
}

// Voice recording
function startVoiceRecording() {
  if (isRequestActive || isRecording) return;
  isRecording = true;
  if (uiElements.sendBtn) uiElements.sendBtn.classList.add('recording');
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.start();
    mediaRecorder.addEventListener('dataavailable', event => audioChunks.push(event.data));
  }).catch(err => {
    console.error('Error accessing microphone:', err);
    alert('فشل الوصول إلى الميكروفون. من فضلك، تحقق من الأذونات.');
    isRecording = false;
    if (uiElements.sendBtn) uiElements.sendBtn.classList.remove('recording');
  });
}

function stopVoiceRecording() {
  if (mediaRecorder?.state === 'recording') {
    mediaRecorder.stop();
    if (uiElements.sendBtn) uiElements.sendBtn.classList.remove('recording');
    isRecording = false;
    mediaRecorder.addEventListener('stop', async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const formData = new FormData();
      formData.append('file', audioBlob, 'voice-message.webm');
      await submitAudioMessage(formData);
    });
  }
}

// Send audio message
async function submitAudioMessage(formData) {
  enterChatView();
  addMsg('user', 'رسالة صوتية');
  conversationHistory.push({ role: 'user', content: 'رسالة صوتية' });
  sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
  streamMsg = addMsg('assistant', '');
  const loadingEl = document.createElement('span');
  loadingEl.className = 'loading';
  streamMsg.appendChild(loadingEl);
  updateUIForRequest();

  isRequestActive = true;
  abortController = new AbortController();

  try {
    const response = await sendRequest('/api/audio-transcription', formData);
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    const data = await response.json();
    if (!data.transcription) throw new Error('لم يتم استلام نص النسخ من الخادم');
    const transcription = data.transcription || 'خطأ: لم يتم إنشاء نص النسخ.';
    if (streamMsg) {
      streamMsg.dataset.text = transcription;
      renderMarkdown(streamMsg);
      streamMsg.dataset.done = '1';
    }
    conversationHistory.push({ role: 'assistant', content: transcription });
    sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
    if (checkAuth() && currentConversationId) {
      await saveMessageToConversation(currentConversationId, 'assistant', transcription);
    }
    if (checkAuth() && data.conversation_id) {
      currentConversationId = data.conversation_id;
      currentConversationTitle = data.conversation_title || 'محادثة بدون عنوان';
      if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = currentConversationTitle;
      history.pushState(null, '', `/chat/${currentConversationId}`);
      await loadConversations();
    }
    finalizeRequest();
  } catch (error) {
    handleRequestError(error);
  }
}

// Helper to send API requests
async function sendRequest(endpoint, body, headers = {}) {
  const token = checkAuth();
  if (token) headers['Authorization'] = `Bearer ${token}`;
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      body,
      headers,
      signal: abortController?.signal,
    });
    if (!response.ok) {
      if (response.status === 403) {
        if (uiElements.messageLimitWarning) uiElements.messageLimitWarning.classList.remove('hidden');
        throw new Error('تم الوصول إلى الحد الأقصى للرسائل. من فضلك، سجل الدخول للمتابعة.');
      }
      if (response.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/login';
        throw new Error('غير مصرح. من فضلك، سجل الدخول مرة أخرى.');
      }
      if (response.status === 503) {
        throw new Error('النموذج غير متاح حاليًا. من فضلك، حاول استخدام نموذج آخر.');
      }
      throw new Error(`فشل الطلب: ${response.status}`);
    }
    return response;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('تم إلغاء الطلب');
    }
    throw error;
  }
}

// Helper to update UI during request
function updateUIForRequest() {
  if (uiElements.stopBtn) uiElements.stopBtn.style.display = 'inline-flex';
  if (uiElements.sendBtn) uiElements.sendBtn.style.display = 'none';
  if (uiElements.input) uiElements.input.value = '';
  if (uiElements.sendBtn) uiElements.sendBtn.disabled = true;
  if (uiElements.filePreview) uiElements.filePreview.style.display = 'none';
  if (uiElements.audioPreview) uiElements.audioPreview.style.display = 'none';
  autoResizeTextarea();
}

// Helper to finalize request
function finalizeRequest() {
  streamMsg = null;
  isRequestActive = false;
  abortController = null;
  if (uiElements.sendBtn) uiElements.sendBtn.style.display = 'inline-flex';
  if (uiElements.stopBtn) uiElements.stopBtn.style.display = 'none';
}

// Helper to handle request errors
function handleRequestError(error) {
  if (streamMsg) {
    streamMsg.querySelector('.loading')?.remove();
    streamMsg.dataset.text = `خطأ: ${error.message || 'حدث خطأ أثناء الطلب.'}`;
    renderMarkdown(streamMsg);
    streamMsg.dataset.done = '1';
    streamMsg = null;
  }
  console.error('خطأ في الطلب:', error);
  alert(`خطأ: ${error.message || 'حدث خطأ أثناء الطلب.'}`);
  isRequestActive = false;
  abortController = null;
  if (uiElements.sendBtn) uiElements.sendBtn.style.display = 'inline-flex';
  if (uiElements.stopBtn) uiElements.stopBtn.style.display = 'none';
}

// Load conversations for sidebar
async function loadConversations() {
  if (!checkAuth()) return;
  try {
    const response = await fetch('/api/conversations', {
      headers: { 'Authorization': `Bearer ${checkAuth()}` }
    });
    if (!response.ok) throw new Error('فشل تحميل المحادثات');
    const conversations = await response.json();
    if (uiElements.conversationList) {
      uiElements.conversationList.innerHTML = '';
      conversations.forEach(conv => {
        const li = document.createElement('li');
        li.className = `flex items-center justify-between text-white hover:bg-gray-700 p-2 rounded cursor-pointer transition-colors ${conv.conversation_id === currentConversationId ? 'bg-gray-700' : ''}`;
        li.dataset.conversationId = conv.conversation_id;
        li.innerHTML = `
          <div class="flex items-center flex-1" style="direction: ${isArabicText(conv.title) ? 'rtl' : 'ltr'};" data-conversation-id="${conv.conversation_id}">
            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path>
            </svg>
            <span class="truncate flex-1">${conv.title || 'محادثة بدون عنوان'}</span>
          </div>
          <button class="delete-conversation-btn text-red-400 hover:text-red-600 p-1" title="حذف المحادثة" data-conversation-id="${conv.conversation_id}">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5-4h4M3 7h18"></path>
            </svg>
          </button>
        `;
        li.querySelector('[data-conversation-id]').addEventListener('click', () => loadConversation(conv.conversation_id));
        li.querySelector('.delete-conversation-btn').addEventListener('click', () => deleteConversation(conv.conversation_id));
        uiElements.conversationList.appendChild(li);
      });
    }
  } catch (error) {
    console.error('خطأ في تحميل المحادثات:', error);
    alert('فشل تحميل المحادثات. من فضلك، حاول مرة أخرى.');
  }
}

// Load conversation from API
async function loadConversation(conversationId) {
  try {
    const response = await fetch(`/api/conversations/${conversationId}`, {
      headers: { 'Authorization': `Bearer ${checkAuth()}` }
    });
    if (!response.ok) {
      if (response.status === 401) window.location.href = '/login';
      throw new Error('فشل تحميل المحادثة');
    }
    const data = await response.json();
    currentConversationId = data.conversation_id;
    currentConversationTitle = data.title || 'محادثة بدون عنوان';
    conversationHistory = data.messages.map(msg => ({ role: msg.role, content: msg.content }));
    if (uiElements.chatBox) uiElements.chatBox.innerHTML = '';
    conversationHistory.forEach(msg => addMsg(msg.role, msg.content));
    enterChatView();
    if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = currentConversationTitle;
    history.pushState(null, '', `/chat/${currentConversationId}`);
    toggleSidebar(false);
  } catch (error) {
    console.error('خطأ في تحميل المحادثة:', error);
    alert('فشل تحميل المحادثة. من فضلك، حاول مرة أخرى أو سجل الدخول.');
  }
}

// Delete conversation
async function deleteConversation(conversationId) {
  if (!confirm('هل أنت متأكد من حذف هذه المحادثة؟')) return;
  try {
    const response = await fetch(`/api/conversations/${conversationId}`, {
      method: 'DELETE',
      headers: { 'Authorization': `Bearer ${checkAuth()}` }
    });
    if (!response.ok) {
      if (response.status === 401) window.location.href = '/login';
      throw new Error('فشل حذف المحادثة');
    }
    if (conversationId === currentConversationId) {
      clearAllMessages();
      currentConversationId = null;
      currentConversationTitle = null;
      history.pushState(null, '', '/chat');
    }
    await loadConversations();
  } catch (error) {
    console.error('خطأ في حذف المحادثة:', error);
    alert('فشل حذف المحادثة. من فضلك، حاول مرة أخرى.');
  }
}

// Save message to conversation
async function saveMessageToConversation(conversationId, role, content) {
  try {
    const response = await fetch(`/api/conversations/${conversationId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${checkAuth()}`
      },
      body: JSON.stringify({ role, content })
    });
    if (!response.ok) throw new Error('فشل حفظ الرسالة');
  } catch (error) {
    console.error('خطأ في حفظ الرسالة:', error);
  }
}

// Create new conversation
async function createNewConversation() {
  if (!checkAuth()) {
    alert('من فضلك، سجل الدخول لإنشاء محادثة جديدة.');
    window.location.href = '/login';
    return;
  }
  try {
    const response = await fetch('/api/conversations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${checkAuth()}`
      },
      body: JSON.stringify({ title: 'محادثة جديدة' })
    });
    if (!response.ok) {
      if (response.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/login';
      }
      throw new Error('فشل إنشاء المحادثة');
    }
    const data = await response.json();
    currentConversationId = data.conversation_id;
    currentConversationTitle = data.title;
    conversationHistory = [];
    sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
    if (uiElements.chatBox) uiElements.chatBox.innerHTML = '';
    if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = currentConversationTitle;
    history.pushState(null, '', `/chat/${currentConversationId}`);
    enterChatView();
    await loadConversations();
    toggleSidebar(false);
  } catch (error) {
    console.error('خطأ في إنشاء المحادثة:', error);
    alert('فشل إنشاء محادثة جديدة. من فضلك، حاول مرة أخرى.');
  }
}

// Update conversation title
async function updateConversationTitle(conversationId, newTitle) {
  try {
    const response = await fetch(`/api/conversations/${conversationId}/title`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${checkAuth()}`
      },
      body: JSON.stringify({ title: newTitle })
    });
    if (!response.ok) throw new Error('فشل تحديث العنوان');
    const data = await response.json();
    currentConversationTitle = data.title;
    if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = currentConversationTitle;
    await loadConversations();
  } catch (error) {
    console.error('خطأ في تحديث العنوان:', error);
    alert('فشل تحديث عنوان المحادثة.');
  }
}

// Toggle sidebar
function toggleSidebar(show) {
  if (uiElements.sidebar) {
    if (window.innerWidth >= 768) {
      isSidebarOpen = true;
      uiElements.sidebar.style.transform = 'translateX(0)';
      if (uiElements.swipeHint) uiElements.swipeHint.style.display = 'none';
    } else {
      isSidebarOpen = show !== undefined ? show : !isSidebarOpen;
      uiElements.sidebar.style.transform = isSidebarOpen ? 'translateX(0)' : 'translateX(-100%)';
      if (uiElements.swipeHint && !isSidebarOpen) {
        uiElements.swipeHint.style.display = 'block';
        setTimeout(() => {
          uiElements.swipeHint.style.display = 'none';
        }, 3000);
      } else if (uiElements.swipeHint) {
        uiElements.swipeHint.style.display = 'none';
      }
    }
  }
}

// Setup touch gestures with Hammer.js
function setupTouchGestures() {
  if (!uiElements.sidebar) return;
  const hammer = new Hammer(uiElements.sidebar);
  const mainContent = document.querySelector('.flex-1');
  const hammerMain = new Hammer(mainContent);

  hammer.get('pan').set({ direction: Hammer.DIRECTION_HORIZONTAL });
  hammer.on('pan', e => {
    if (!isSidebarOpen) return;
    let translateX = Math.max(-uiElements.sidebar.offsetWidth, Math.min(0, e.deltaX));
    uiElements.sidebar.style.transform = `translateX(${translateX}px)`;
  });
  hammer.on('panend', e => {
    if (!isSidebarOpen) return;
    if (e.deltaX < -50) {
      toggleSidebar(false);
    } else {
      toggleSidebar(true);
    }
  });

  hammerMain.get('pan').set({ direction: Hammer.DIRECTION_HORIZONTAL });
  hammerMain.on('panstart', e => {
    if (isSidebarOpen) return;
    if (e.center.x < 50 || e.center.x > window.innerWidth - 50) {
      uiElements.sidebar.style.transition = 'none';
    }
  });
  hammerMain.on('pan', e => {
    if (isSidebarOpen) return;
    if (e.center.x < 50 || e.center.x > window.innerWidth - 50) {
      let translateX = e.center.x < 50
        ? Math.min(uiElements.sidebar.offsetWidth, Math.max(0, e.deltaX))
        : Math.max(-uiElements.sidebar.offsetWidth, Math.min(0, e.deltaX));
      uiElements.sidebar.style.transform = `translateX(${translateX - uiElements.sidebar.offsetWidth}px)`;
    }
  });
  hammerMain.on('panend', e => {
    uiElements.sidebar.style.transition = 'transform 0.3s ease-in-out';
    if (e.center.x < 50 && e.deltaX > 50) {
      toggleSidebar(true);
    } else if (e.center.x > window.innerWidth - 50 && e.deltaX < -50) {
      toggleSidebar(true);
    } else {
      toggleSidebar(false);
    }
  });
}

// Send user message
async function submitMessage() {
  if (isRequestActive || isRecording) return;
  let message = uiElements.input?.value.trim() || '';
  let payload = null;
  let formData = null;
  let endpoint = '/api/chat';
  let headers = {};
  let inputType = 'text';
  let outputFormat = 'text';
  let title = null;

  if (uiElements.fileInput?.files.length > 0) {
    const file = uiElements.fileInput.files[0];
    if (file.type.startsWith('image/')) {
      endpoint = '/api/image-analysis';
      inputType = 'image';
      message = 'تحليل هذه الصورة';
      formData = new FormData();
      formData.append('file', file);
      formData.append('output_format', 'text');
    }
  } else if (uiElements.audioInput?.files.length > 0) {
    const file = uiElements.audioInput.files[0];
    if (file.type.startsWith('audio/')) {
      endpoint = '/api/audio-transcription';
      inputType = 'audio';
      message = 'نسخ هذا الصوت';
      formData = new FormData();
      formData.append('file', file);
    }
  } else if (message) {
    payload = {
      message,
      system_prompt: isArabicText(message)
        ? 'أنت مساعد ذكي تقدم إجابات مفصلة ومنظمة باللغة العربية، مع ضمان الدقة والوضوح.'
        : 'You are an expert assistant providing detailed, comprehensive, and well-structured responses.',
      history: conversationHistory,
      temperature: 0.7,
      max_new_tokens: 128000,
      enable_browsing: true,
      output_format: 'text',
      title: title
    };
    headers['Content-Type'] = 'application/json';
  } else {
    return;
  }

  enterChatView();
  addMsg('user', message);
  conversationHistory.push({ role: 'user', content: message });
  sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
  streamMsg = addMsg('assistant', '');
  const loadingEl = document.createElement('span');
  loadingEl.className = 'loading';
  streamMsg.appendChild(loadingEl);
  updateUIForRequest();

  isRequestActive = true;
  abortController = new AbortController();

  try {
    const response = await sendRequest(endpoint, payload ? JSON.stringify(payload) : formData, headers);
    if (endpoint === '/api/audio-transcription') {
      const data = await response.json();
      if (!data.transcription) throw new Error('لم يتم استلام نص النسخ من الخادم');
      const transcription = data.transcription || 'خطأ: لم يتم إنشاء نص النسخ.';
      if (streamMsg) {
        streamMsg.dataset.text = transcription;
        renderMarkdown(streamMsg);
        streamMsg.dataset.done = '1';
      }
      conversationHistory.push({ role: 'assistant', content: transcription });
      sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
      if (checkAuth() && currentConversationId) {
        await saveMessageToConversation(currentConversationId, 'assistant', transcription);
      }
      if (checkAuth() && data.conversation_id) {
        currentConversationId = data.conversation_id;
        currentConversationTitle = data.conversation_title || 'محادثة بدون عنوان';
        if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = currentConversationTitle;
        history.pushState(null, '', `/chat/${currentConversationId}`);
        await loadConversations();
      }
    } else if (endpoint === '/api/image-analysis') {
      const data = await response.json();
      const analysis = data.image_analysis || 'خطأ: لم يتم إنشاء تحليل.';
      if (streamMsg) {
        streamMsg.dataset.text = analysis;
        renderMarkdown(streamMsg);
        streamMsg.dataset.done = '1';
      }
      conversationHistory.push({ role: 'assistant', content: analysis });
      sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
      if (checkAuth() && currentConversationId) {
        await saveMessageToConversation(currentConversationId, 'assistant', analysis);
      }
      if (checkAuth() && data.conversation_id) {
        currentConversationId = data.conversation_id;
        currentConversationTitle = data.conversation_title || 'محادثة بدون عنوان';
        if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = currentConversationTitle;
        history.pushState(null, '', `/chat/${currentConversationId}`);
        await loadConversations();
      }
    } else {
      const contentType = response.headers.get('Content-Type');
      if (contentType?.includes('application/json')) {
        const data = await response.json();
        const responseText = data.response || 'خطأ: لم يتم إنشاء رد.';
        if (streamMsg) {
          streamMsg.dataset.text = responseText;
          renderMarkdown(streamMsg);
          streamMsg.dataset.done = '1';
        }
        conversationHistory.push({ role: 'assistant', content: responseText });
        sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
        if (checkAuth() && currentConversationId) {
          await saveMessageToConversation(currentConversationId, 'assistant', responseText);
        }
        if (checkAuth() && data.conversation_id) {
          currentConversationId = data.conversation_id;
          currentConversationTitle = data.conversation_title || 'محادثة بدون عنوان';
          if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = currentConversationTitle;
          history.pushState(null, '', `/chat/${currentConversationId}`);
          await loadConversations();
        }
      } else {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            if (!buffer.trim()) throw new Error('الرد فارغ من الخادم');
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          if (streamMsg) {
            streamMsg.dataset.text = buffer;
            currentAssistantText = buffer;
            streamMsg.querySelector('.loading')?.remove();
            renderMarkdown(streamMsg);
            if (uiElements.chatBox) uiElements.chatBox.scrollTop = uiElements.chatBox.scrollHeight;
          }
        }
        if (streamMsg) streamMsg.dataset.done = '1';
        conversationHistory.push({ role: 'assistant', content: buffer });
        sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
        if (checkAuth() && currentConversationId) {
          await saveMessageToConversation(currentConversationId, 'assistant', buffer);
        }
      }
    }
    finalizeRequest();
  } catch (error) {
    handleRequestError(error);
  }
}

// Stop streaming
function stopStream(forceCancel = false) {
  if (!isRequestActive && !isRecording) return;
  if (isRecording) stopVoiceRecording();
  isRequestActive = false;
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
  if (streamMsg && !forceCancel) {
    streamMsg.querySelector('.loading')?.remove();
    streamMsg.dataset.text += '';
    renderMarkdown(streamMsg);
    streamMsg.dataset.done = '1';
    streamMsg = null;
  }
  if (uiElements.stopBtn) uiElements.stopBtn.style.display = 'none';
  if (uiElements.sendBtn) uiElements.sendBtn.style.display = 'inline-flex';
  if (uiElements.stopBtn) uiElements.stopBtn.style.pointerEvents = 'auto';
}

// Settings Modal
if (uiElements.settingsBtn) {
  uiElements.settingsBtn.addEventListener('click', () => {
    if (!checkAuth()) {
      alert('من فضلك، سجل الدخول للوصول إلى الإعدادات.');
      window.location.href = '/login';
      return;
    }
    uiElements.settingsModal.classList.remove('hidden');
    fetch('/api/settings', {
      headers: { 'Authorization': `Bearer ${checkAuth()}` }
    })
      .then(res => {
        if (!res.ok) {
          if (res.status === 401) {
            localStorage.removeItem('token');
            window.location.href = '/login';
          }
          throw new Error('فشل جلب الإعدادات');
        }
        return res.json();
      })
      .then(data => {
        document.getElementById('display_name').value = data.user_settings.display_name || '';
        document.getElementById('preferred_model').value = data.user_settings.preferred_model || 'standard';
        document.getElementById('job_title').value = data.user_settings.job_title || '';
        document.getElementById('education').value = data.user_settings.education || '';
        document.getElementById('interests').value = data.user_settings.interests || '';
        document.getElementById('additional_info').value = data.user_settings.additional_info || '';
        document.getElementById('conversation_style').value = data.user_settings.conversation_style || 'default';

        const modelSelect = document.getElementById('preferred_model');
        modelSelect.innerHTML = '';
        data.available_models.forEach(model => {
          const option = document.createElement('option');
          option.value = model.alias;
          option.textContent = `${model.alias} - ${model.description}`;
          modelSelect.appendChild(option);
        });

        const styleSelect = document.getElementById('conversation_style');
        styleSelect.innerHTML = '';
        data.conversation_styles.forEach(style => {
          const option = document.createElement('option');
          option.value = style;
          option.textContent = style.charAt(0).toUpperCase() + style.slice(1);
          styleSelect.appendChild(option);
        });
      })
      .catch(err => {
        console.error('خطأ في جلب الإعدادات:', err);
        alert('فشل تحميل الإعدادات. من فضلك، حاول مرة أخرى.');
      });
  });
}

if (uiElements.closeSettingsBtn) {
  uiElements.closeSettingsBtn.addEventListener('click', () => {
    uiElements.settingsModal.classList.add('hidden');
  });
}

if (uiElements.settingsForm) {
  uiElements.settingsForm.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!checkAuth()) {
      alert('من فضلك، سجل الدخول لحفظ الإعدادات.');
      window.location.href = '/login';
      return;
    }
    const formData = new FormData(uiElements.settingsForm);
    const data = Object.fromEntries(formData);
    fetch('/users/me', {
      method: 'PUT',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${checkAuth()}`
      },
      body: JSON.stringify(data)
    })
      .then(res => {
        if (!res.ok) {
          if (res.status === 401) {
            localStorage.removeItem('token');
            window.location.href = '/login';
          }
          throw new Error('فشل تحديث الإعدادات');
        }
        return res.json();
      })
      .then(() => {
        alert('تم تحديث الإعدادات بنجاح!');
        uiElements.settingsModal.classList.add('hidden');
        toggleSidebar(false);
      })
      .catch(err => {
        console.error('خطأ في تحديث الإعدادات:', err);
        alert('خطأ في تحديث الإعدادات: ' + err.message);
      });
  });
}

// History Toggle
if (uiElements.historyToggle) {
  uiElements.historyToggle.addEventListener('click', () => {
    if (uiElements.conversationList) {
      uiElements.conversationList.classList.toggle('hidden');
      uiElements.historyToggle.innerHTML = uiElements.conversationList.classList.contains('hidden')
        ? `<svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
           </svg>إظهار السجل`
        : `<svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
           </svg>إخفاء السجل`;
    }
  });
}

// Event listeners
uiElements.promptItems.forEach(p => {
  p.addEventListener('click', e => {
    e.preventDefault();
    if (uiElements.input) {
      uiElements.input.value = p.dataset.prompt;
      autoResizeTextarea();
    }
    if (uiElements.sendBtn) uiElements.sendBtn.disabled = false;
    submitMessage();
  });
});

if (uiElements.fileBtn) uiElements.fileBtn.addEventListener('click', () => uiElements.fileInput?.click());
if (uiElements.audioBtn) uiElements.audioBtn.addEventListener('click', () => uiElements.audioInput?.click());
if (uiElements.fileInput) uiElements.fileInput.addEventListener('change', previewFile);
if (uiElements.audioInput) uiElements.audioInput.addEventListener('change', previewFile);

if (uiElements.sendBtn) {
  uiElements.sendBtn.addEventListener('mousedown', e => {
    if (uiElements.sendBtn.disabled || isRequestActive || isRecording) return;
    startVoiceRecording();
  });
  uiElements.sendBtn.addEventListener('mouseup', () => isRecording && stopVoiceRecording());
  uiElements.sendBtn.addEventListener('mouseleave', () => isRecording && stopVoiceRecording());
  uiElements.sendBtn.addEventListener('touchstart', e => {
    e.preventDefault();
    if (uiElements.sendBtn.disabled || isRequestActive || isRecording) return;
    startVoiceRecording();
  });
  uiElements.sendBtn.addEventListener('touchend', e => {
    e.preventDefault();
    if (isRecording) stopVoiceRecording();
  });
  uiElements.sendBtn.addEventListener('touchcancel', e => {
    e.preventDefault();
    if (isRecording) stopVoiceRecording();
  });
}

if (uiElements.form) {
  uiElements.form.addEventListener('submit', e => {
    e.preventDefault();
    if (!isRecording) submitMessage();
  });
}

if (uiElements.input) {
  uiElements.input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isRecording && !uiElements.sendBtn.disabled) submitMessage();
    }
  });
  uiElements.input.addEventListener('input', () => {
    autoResizeTextarea();
    updateSendButtonState();
  });
}

if (uiElements.stopBtn) {
  uiElements.stopBtn.addEventListener('click', () => {
    uiElements.stopBtn.style.pointerEvents = 'none';
    stopStream();
  });
}

if (uiElements.clearBtn) uiElements.clearBtn.addEventListener('click', clearAllMessages);

if (uiElements.conversationTitle) {
  uiElements.conversationTitle.addEventListener('click', () => {
    if (!checkAuth()) return alert('من فضلك، سجل الدخول لتعديل عنوان المحادثة.');
    const newTitle = prompt('أدخل عنوان المحادثة الجديد:', currentConversationTitle || '');
    if (newTitle && currentConversationId) {
      updateConversationTitle(currentConversationId, newTitle);
    }
  });
}

if (uiElements.sidebarToggle) {
  uiElements.sidebarToggle.addEventListener('click', () => toggleSidebar());
}

if (uiElements.newConversationBtn) {
  uiElements.newConversationBtn.addEventListener('click', createNewConversation);
}
