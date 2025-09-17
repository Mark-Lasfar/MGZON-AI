// SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
// SPDX-License-Identifier: Apache-2.0

// Prism for code highlighting
Prism.plugins.autoloader.languages_path = 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/';

// UI elements with fallback check
const uiElements = {
  chatArea: document.getElementById('chatArea') || document.createElement('div'),
  chatBox: document.getElementById('chatBox') || document.createElement('div'),
  initialContent: document.getElementById('initialContent') || document.createElement('div'),
  form: document.getElementById('footerForm') || document.createElement('form'),
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
  historyToggle: document.getElementById('historyToggle'),
};

// State variables
let conversationHistory = JSON.parse(sessionStorage.getItem('conversationHistory') || '[]');
let currentConversationId = window.conversationId || null;
let currentConversationTitle = window.conversationTitle || null;
let isRequestActive = false;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let streamMsg = null;
let currentAssistantText = '';
let isSidebarOpen = window.innerWidth >= 768;
let abortController = null;

// Forcefully initialize chat view on load
document.addEventListener('DOMContentLoaded', async () => {
  AOS.init({
    duration: 800,
    easing: 'ease-out-cubic',
    once: true,
    offset: 50,
  });

  // Force chat view to be visible immediately
  enterChatView(true);

  if (await checkAuth() && currentConversationId) {
    console.log('Loading conversation with ID:', currentConversationId);
    await loadConversation(currentConversationId);
  } else if (!(await checkAuth()) && conversationHistory.length > 0) {
    console.log('Restoring conversation history from sessionStorage:', conversationHistory);
    conversationHistory.forEach(msg => {
      console.log('Adding message from history:', msg);
      addMsg(msg.role, msg.content);
    });
  } else {
    console.log('No conversation history or ID, starting fresh');
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
async function checkAuth() {
  const token = localStorage.getItem('token');
  if (!token) {
    console.log('No auth token found');
    return false;
  }
  try {
    const response = await fetch('/api/verify-token', {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (response.ok) {
      console.log('Auth token verified');
      return true;
    } else {
      console.log('Token verification failed:', response.status);
      localStorage.removeItem('token');
      return false;
    }
  } catch (error) {
    console.error('Error verifying token:', error);
    localStorage.removeItem('token');
    return false;
  }
}

// Handle session for non-logged-in users
async function handleSession() {
  const sessionId = sessionStorage.getItem('session_id');
  if (!sessionId) {
    const newSessionId = crypto.randomUUID();
    sessionStorage.setItem('session_id', newSessionId);
    console.log('New session_id created:', newSessionId);
    return newSessionId;
  }
  console.log('Existing session_id:', sessionId);
  return sessionId;
}

// Update send button state
function updateSendButtonState() {
  if (uiElements.sendBtn && uiElements.input && uiElements.fileInput && uiElements.audioInput) {
    const hasInput = uiElements.input.value.trim() !== '' ||
                     uiElements.fileInput.files.length > 0 ||
                     uiElements.audioInput.files.length > 0;
    uiElements.sendBtn.disabled = !hasInput || isRequestActive || isRecording;
    console.log('Send button state updated:', { hasInput, isRequestActive, isRecording, disabled: uiElements.sendBtn.disabled });
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
  // إضافة زر نسخ لكل بلوك كود
  wrapper.querySelectorAll('pre').forEach(pre => {
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-btn';
    copyBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>';
    copyBtn.title = 'Copy Code';
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(pre.querySelector('code').innerText).then(() => {
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>'; }, 2000);
      });
    };
    pre.appendChild(copyBtn);
  });

  wrapper.querySelectorAll('hr').forEach(h => h.classList.add('styled-hr'));
  Prism.highlightAllUnder(wrapper);
  if (uiElements.chatBox) {
    uiElements.chatBox.scrollTo({
      top: uiElements.chatBox.scrollHeight,
      behavior: 'smooth',
    });
  }
  el.style.display = 'block';
}

// Toggle chat view with force option
function enterChatView(force = false) {
  if (uiElements.chatHeader) {
    uiElements.chatHeader.classList.remove('hidden');
    uiElements.chatHeader.setAttribute('aria-hidden', 'false');
    if (currentConversationTitle && uiElements.conversationTitle) {
      uiElements.conversationTitle.textContent = currentConversationTitle;
    }
  }
  if (uiElements.chatArea) {
    uiElements.chatArea.classList.remove('hidden');
    uiElements.chatArea.style.display = force ? 'flex !important' : 'flex';
    uiElements.chatArea.style.opacity = '1';
    uiElements.chatArea.style.visibility = 'visible';
  }
  if (uiElements.chatBox) {
    uiElements.chatBox.classList.remove('hidden');
    uiElements.chatBox.style.display = force ? 'flex !important' : 'flex';
    uiElements.chatBox.style.opacity = '1';
    uiElements.chatBox.style.visibility = 'visible';
  }
  if (uiElements.initialContent) uiElements.initialContent.classList.add('hidden');
  if (uiElements.form) {
    uiElements.form.classList.remove('hidden');
    uiElements.form.style.display = force ? 'flex !important' : 'flex';
    uiElements.form.style.opacity = '1';
    uiElements.form.style.visibility = 'visible';
  }
  console.log('Chat view forced to enter:', {
    chatArea: uiElements.chatArea?.style.display,
    chatBox: uiElements.chatBox?.style.display,
    form: uiElements.form?.style.display
  });
}

// Toggle home view
function leaveChatView() {
  if (uiElements.chatHeader) {
    uiElements.chatHeader.classList.add('hidden');
    uiElements.chatHeader.setAttribute('aria-hidden', 'true');
  }
  if (uiElements.chatBox) uiElements.chatBox.classList.add('hidden');
  if (uiElements.initialContent) uiElements.initialContent.classList.remove('hidden');
  if (uiElements.form) uiElements.form.classList.add('hidden');
}

// Add chat bubble
function addMsg(who, text) {
  const container = document.createElement('div');
  container.className = 'message-container';
  const div = document.createElement('div');
  div.className = `bubble ${who === 'user' ? 'bubble-user' : 'bubble-assist'} ${isArabicText(text) ? 'rtl' : ''}`;
  div.dataset.text = text;
  console.log('Adding message:', { who, text });
  renderMarkdown(div);
  div.style.display = 'block';

  // إضافة أيقونات التحكم
  const actions = document.createElement('div');
  actions.className = 'message-actions';

  // زر نسخ الرد
  const copyBtn = document.createElement('button');
  copyBtn.className = 'action-btn';
  copyBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>';
  copyBtn.title = 'Copy Response';
  copyBtn.onclick = () => {
    navigator.clipboard.writeText(text).then(() => {
      copyBtn.textContent = 'Copied!';
      setTimeout(() => { copyBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>'; }, 2000);
    });
  };
  actions.appendChild(copyBtn);

  // زر إعادة المحاولة (للمساعد فقط)
  if (who === 'assistant') {
    const retryBtn = document.createElement('button');
    retryBtn.className = 'action-btn';
    retryBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>';
    retryBtn.title = 'Retry';
    retryBtn.onclick = () => submitMessage(); // إعادة إرسال السؤال
    actions.appendChild(retryBtn);
  }

  // زر تعديل السؤال (للمستخدم فقط)
  if (who === 'user') {
    const editBtn = document.createElement('button');
    editBtn.className = 'action-btn';
    editBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>';
    editBtn.title = 'Edit Question';
    editBtn.onclick = () => {
      const newText = prompt('Edit your question:', text);
      if (newText) {
        div.dataset.text = newText;
        renderMarkdown(div);
        conversationHistory = conversationHistory.map(msg => msg.role === 'user' && msg.content === text ? { role: 'user', content: newText } : msg);
        sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
      }
    };
    actions.appendChild(editBtn);
  }

  container.appendChild(div);
  container.appendChild(actions);
  if (uiElements.chatBox) {
    uiElements.chatBox.appendChild(container);
    uiElements.chatBox.scrollTop = uiElements.chatBox.scrollHeight;
  } else {
    console.error('chatBox not found, appending to a fallback container');
    document.body.appendChild(container);
  }
  return div;
}

// Clear all messages
function clearAllMessages() {
  stopStream(true);
  conversationHistory = [];
  sessionStorage.removeItem('conversationHistory');
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
  if (isRequestActive || isRecording) {
    console.log('Voice recording blocked: Request active or already recording');
    return;
  }
  console.log('Starting voice recording...');
  isRecording = true;
  if (uiElements.sendBtn) uiElements.sendBtn.classList.add('recording');
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.start();
    console.log('MediaRecorder started');
    mediaRecorder.addEventListener('dataavailable', event => {
      audioChunks.push(event.data);
      console.log('Audio chunk received:', event.data);
    });
  }).catch(err => {
    console.error('Error accessing microphone:', err);
    alert('Failed to access microphone. Please check permissions.');
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
      console.log('Stopping voice recording, sending audio...');
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
  addMsg('user', 'Voice message');
  if (!(await checkAuth())) {
    conversationHistory.push({ role: 'user', content: 'Voice message' });
    sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
  }
  streamMsg = addMsg('assistant', '');
  const loadingEl = document.createElement('span');
  loadingEl.className = 'loading';
  streamMsg.appendChild(loadingEl);
  updateUIForRequest();

  isRequestActive = true;
  abortController = new AbortController();

  try {
    const response = await sendRequest('/api/audio-transcription', formData);
    if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
    const data = await response.json();
    if (!data.transcription) throw new Error('No transcription received from server');
    const transcription = data.transcription || 'Error: No transcription generated.';
    if (streamMsg) {
      streamMsg.dataset.text = transcription;
      renderMarkdown(streamMsg);
      streamMsg.dataset.done = '1';
    }
    if (!(await checkAuth())) {
      conversationHistory.push({ role: 'assistant', content: transcription });
      sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
    }
    if (checkAuth() && data.conversation_id) {
      currentConversationId = data.conversation_id;
      currentConversationTitle = data.conversation_title || 'Untitled Conversation';
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
  const token = localStorage.getItem('token');
  if (token) headers['Authorization'] = `Bearer ${token}`;
  headers['X-Session-ID'] = await handleSession();
  console.log('Sending request to:', endpoint, 'with headers:', headers);
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
        throw new Error('Message limit reached. Please log in to continue.');
      }
      if (response.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/login';
        throw new Error('Unauthorized. Please log in again.');
      }
      if (response.status === 503) {
        throw new Error('Model not available. Please try another model.');
      }
      throw new Error(`Request failed with status ${response.status}`);
    }
    return response;
  } catch (error) {
    console.error('Send request error:', error);
    if (error.name === 'AbortError') {
      throw new Error('Request was aborted');
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
  if (uiElements.sendBtn) {
    uiElements.sendBtn.style.display = 'inline-flex';
    uiElements.sendBtn.disabled = false;
  }
  if (uiElements.stopBtn) uiElements.stopBtn.style.display = 'none';
  updateSendButtonState();
}

// Helper to handle request errors
function handleRequestError(error) {
  if (streamMsg) {
    streamMsg.querySelector('.loading')?.remove();
    streamMsg.dataset.text = `Error: ${error.message || 'An error occurred during the request.'}`;
    const retryBtn = document.createElement('button');
    retryBtn.innerText = 'Retry';
    retryBtn.className = 'retry-btn text-sm text-blue-400 hover:text-blue-600';
    retryBtn.onclick = () => submitMessage();
    streamMsg.appendChild(retryBtn);
    renderMarkdown(streamMsg);
    streamMsg.dataset.done = '1';
    streamMsg = null;
  }
  console.error('Request error:', error);
  alert(`Error: ${error.message || 'An error occurred during the request.'}`);
  isRequestActive = false;
  abortController = null;
  if (!(checkAuth())) {
    sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
  }
  if (uiElements.sendBtn) {
    uiElements.sendBtn.style.display = 'inline-flex';
    uiElements.sendBtn.disabled = false;
  }
  if (uiElements.stopBtn) uiElements.stopBtn.style.display = 'none';
  enterChatView();
}

// Load conversations for sidebar
async function loadConversations() {
  if (!(await checkAuth())) return;
  try {
    const response = await fetch('/api/conversations', {
      headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
    });
    if (!response.ok) throw new Error('Failed to load conversations');
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
            <span class="truncate flex-1">${conv.title || 'Untitled Conversation'}</span>
          </div>
          <button class="delete-conversation-btn text-red-400 hover:text-red-600 p-1" title="Delete Conversation" data-conversation-id="${conv.conversation_id}">
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
    console.error('Error loading conversations:', error);
    alert('Failed to load conversations. Please try again.');
  }
}

// Load conversation from API
async function loadConversation(conversationId) {
  try {
    const response = await fetch(`/api/conversations/${conversationId}`, {
      headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
    });
    if (!response.ok) {
      if (response.status === 401) window.location.href = '/login';
      throw new Error('Failed to load conversation');
    }
    const data = await response.json();
    currentConversationId = data.conversation_id;
    currentConversationTitle = data.title || 'Untitled Conversation';
    conversationHistory = data.messages.map(msg => ({ role: msg.role, content: msg.content }));
    if (uiElements.chatBox) uiElements.chatBox.innerHTML = '';
    conversationHistory.forEach(msg => addMsg(msg.role, msg.content));
    enterChatView();
    if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = currentConversationTitle;
    history.pushState(null, '', `/chat/${currentConversationId}`);
    toggleSidebar(false);
  } catch (error) {
    console.error('Error loading conversation:', error);
    alert('Failed to load conversation. Please try again or log in.');
  }
}

// Delete conversation
async function deleteConversation(conversationId) {
  if (!confirm('Are you sure you want to delete this conversation?')) return;
  try {
    const response = await fetch(`/api/conversations/${conversationId}`, {
      method: 'DELETE',
      headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
    });
    if (!response.ok) {
      if (response.status === 401) window.location.href = '/login';
      throw new Error('Failed to delete conversation');
    }
    if (conversationId === currentConversationId) {
      clearAllMessages();
      currentConversationId = null;
      currentConversationTitle = null;
      history.pushState(null, '', '/chat');
    }
    await loadConversations();
  } catch (error) {
    console.error('Error deleting conversation:', error);
    alert('Failed to delete conversation. Please try again.');
  }
}

// Create new conversation
async function createNewConversation() {
  if (!(await checkAuth())) {
    alert('Please log in to create a new conversation.');
    window.location.href = '/login';
    return;
  }
  try {
    const response = await fetch('/api/conversations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({ title: 'New Conversation' })
    });
    if (!response.ok) {
      if (response.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/login';
      }
      throw new Error('Failed to create conversation');
    }
    const data = await response.json();
    currentConversationId = data.conversation_id;
    currentConversationTitle = data.title;
    conversationHistory = [];
    sessionStorage.removeItem('conversationHistory');
    if (uiElements.chatBox) uiElements.chatBox.innerHTML = '';
    if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = currentConversationTitle;
    history.pushState(null, '', `/chat/${currentConversationId}`);
    enterChatView();
    await loadConversations();
    toggleSidebar(false);
  } catch (error) {
    console.error('Error creating conversation:', error);
    alert('Failed to create new conversation. Please try again.');
  }
  if (uiElements.chatBox) uiElements.chatBox.scrollTo({
    top: uiElements.chatBox.scrollHeight,
    behavior: 'smooth',
  });
}

// Update conversation title
async function updateConversationTitle(conversationId, newTitle) {
  try {
    const response = await fetch(`/api/conversations/${conversationId}/title`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({ title: newTitle })
    });
    if (!response.ok) throw new Error('Failed to update title');
    const data = await response.json();
    currentConversationTitle = data.title;
    if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = currentConversationTitle;
    await loadConversations();
  } catch (error) {
    console.error('Error updating title:', error);
    alert('Failed to update conversation title.');
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
    uiElements.sidebar.style.transition = 'none';
  });
  hammer.on('panend', e => {
    uiElements.sidebar.style.transition = 'transform 0.3s ease-in-out';
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

  if (!message && !uiElements.fileInput?.files.length && !uiElements.audioInput?.files.length) {
    console.log('No message, file, or audio to send');
    return;
  }

  enterChatView(); // دايمًا إظهار المحادثة قبل الإرسال

  if (uiElements.fileInput?.files.length > 0) {
    const file = uiElements.fileInput.files[0];
    if (file.type.startsWith('image/')) {
      endpoint = '/api/image-analysis';
      inputType = 'image';
      message = 'Analyze this image';
      formData = new FormData();
      formData.append('file', file);
      formData.append('output_format', 'text');
    }
  } else if (uiElements.audioInput?.files.length > 0) {
    const file = uiElements.audioInput.files[0];
    if (file.type.startsWith('audio/')) {
      endpoint = '/api/audio-transcription';
      inputType = 'audio';
      message = 'Transcribe this audio';
      formData = new FormData();
      formData.append('file', file);
    }
  } else if (message) {
    payload = {
      message,
      system_prompt: isArabicText(message)
        ? 'أنت مساعد ذكي تقدم إجابات مفصلة ومنظمة باللغة العربية، مع ضمان الدقة والوضوح.'
        : 'You are an expert assistant providing detailed, comprehensive, and well-structured responses.',
      history: (await checkAuth()) ? [] : conversationHistory,
      temperature: 0.7,
      max_new_tokens: 128000,
      enable_browsing: true,
      output_format: 'text'
    };
    headers['Content-Type'] = 'application/json';
  }

  addMsg('user', message);
  if (!(await checkAuth())) {
    conversationHistory.push({ role: 'user', content: message });
    sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
  }
  streamMsg = addMsg('assistant', '');
  const thinkingEl = document.createElement('span');
  thinkingEl.className = 'thinking';
  thinkingEl.textContent = 'The model is thinking...';
  streamMsg.appendChild(thinkingEl);
  updateUIForRequest();

  isRequestActive = true;
  abortController = new AbortController();
  const startTime = Date.now();

  try {
    const response = await sendRequest(endpoint, payload ? JSON.stringify(payload) : formData, headers);
    let responseText = '';
    if (endpoint === '/api/audio-transcription') {
      const data = await response.json();
      if (!data.transcription) throw new Error('No transcription received from server');
      responseText = data.transcription || 'Error: No transcription generated.';
      streamMsg.dataset.text = responseText;
      renderMarkdown(streamMsg);
      streamMsg.dataset.done = '1';
    } else if (endpoint === '/api/image-analysis') {
      const data = await response.json();
      responseText = data.image_analysis || 'Error: No analysis generated.';
      streamMsg.dataset.text = responseText;
      renderMarkdown(streamMsg);
      streamMsg.dataset.done = '1';
    } else {
      const contentType = response.headers.get('Content-Type');
      if (contentType?.includes('application/json')) {
        const data = await response.json();
        responseText = data.response || 'Error: No response generated.';
        if (data.conversation_id) {
          currentConversationId = data.conversation_id;
          currentConversationTitle = data.conversation_title || 'Untitled Conversation';
          if (uiElements.conversationTitle) uiElements.conversationTitle.textContent = currentConversationTitle;
          history.pushState(null, '', `/chat/${currentConversationId}`);
          await loadConversations();
        }
      } else {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        streamMsg.dataset.text = '';
        streamMsg.querySelector('.thinking')?.remove();

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            if (!buffer.trim()) throw new Error('Empty response from server');
            break;
          }
          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;
          console.log('Received chunk:', chunk);

          if (streamMsg) {
            streamMsg.dataset.text = buffer;
            currentAssistantText = buffer;
            renderMarkdown(streamMsg);
            streamMsg.style.opacity = '1';
            if (uiElements.chatBox) {
              uiElements.chatBox.style.display = 'flex';
              uiElements.chatBox.scrollTop = uiElements.chatBox.scrollHeight;
            }
            await new Promise(resolve => setTimeout(resolve, 30)); // تأخير 30ms لتأثير الكتابة الطبيعي
          }
        }
        responseText = buffer;
      }
    }

    const endTime = DateTime.now();
    const thinkingTime = Math.round((endTime - startTime) / 1000); // حساب الوقت بالثواني
    streamMsg.dataset.text += `\n\n*Processed in ${thinkingTime} seconds.*`;
    if (streamMsg) {
      streamMsg.dataset.text = responseText;
      renderMarkdown(streamMsg);
      streamMsg.dataset.done = '1';
    }
    if (!(await checkAuth())) {
      conversationHistory.push({ role: 'assistant', content: responseText });
      sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
    }
    finalizeRequest();
  } catch (error) {
    handleRequestError(error);
  }
}

let attemptCount = 0;
let attempts = []; // لتخزين الردود السابقة

function addAttemptHistory(who, text) {
  attemptCount++;
  attempts.push(text); // تخزين الرد
  const container = document.createElement('div');
  container.className = 'message-container';
  const div = document.createElement('div');
  div.className = `bubble ${who === 'user' ? 'bubble-user' : 'bubble-assist'} ${isArabicText(text) ? 'rtl' : ''}`;
  div.dataset.text = '';
  renderMarkdown(div);

  const historyActions = document.createElement('div');
  historyActions.className = 'message-actions';
  const prevBtn = document.createElement('button');
  prevBtn.className = 'action-btn';
  prevBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M15 19l-7-7 7-7"></path></svg>';
  prevBtn.title = 'Previous Attempt';
  prevBtn.onclick = () => {
    if (attemptCount > 1) {
      attemptCount--;
      div.dataset.text = attempts[attemptCount - 1];
      renderMarkdown(div);
    }
  };
  const nextBtn = document.createElement('button');
  nextBtn.className = 'action-btn';
  nextBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 5l7 7-7 7"></path></svg>';
  nextBtn.title = 'Next Attempt';
  nextBtn.onclick = () => {
    if (attemptCount < attempts.length) {
      attemptCount++;
      div.dataset.text = attempts[attemptCount - 1];
      renderMarkdown(div);
    }
  };
  historyActions.appendChild(prevBtn);
  historyActions.appendChild(document.createTextNode(`Attempt ${attemptCount}`));
  historyActions.appendChild(nextBtn);

  container.appendChild(div);
  container.appendChild(historyActions);
  if (uiElements.chatBox) {
    uiElements.chatBox.appendChild(container);
    uiElements.chatBox.scrollTop = uiElements.chatBox.scrollHeight;
  } else {
    console.error('chatBox not found, appending to a fallback container');
    document.body.appendChild(container);
  }
  return div;
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
  enterChatView();
}

// Logout handler
const logoutBtn = document.querySelector('#logoutBtn');
if (logoutBtn) {
  logoutBtn.addEventListener('click', async () => {
    console.log('Logout button clicked');
    try {
      const response = await fetch('/logout', {
        method: 'POST',
        credentials: 'include'
      });
      if (response.ok) {
        localStorage.removeItem('token');
        console.log('Token removed from localStorage');
        window.location.href = '/login';
      } else {
        console.error('Logout failed:', response.status);
        alert('Failed to log out. Please try again.');
      }
    } catch (error) {
      console.error('Logout error:', error);
      alert('Error during logout: ' + error.message);
    }
  });
}

// Settings Modal
if (uiElements.settingsBtn) {
  uiElements.settingsBtn.addEventListener('click', async () => {
    if (!(await checkAuth())) {
      alert('Please log in to access settings.');
      window.location.href = '/login';
      return;
    }
    try {
      const response = await fetch('/api/settings', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      if (!response.ok) {
        if (response.status === 401) {
          localStorage.removeItem('token');
          window.location.href = '/login';
        }
        throw new Error('Failed to fetch settings');
      }
      const data = await response.json();
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

      uiElements.settingsModal.classList.remove('hidden');
      toggleSidebar(false);
    } catch (err) {
      console.error('Error fetching settings:', err);
      alert('Failed to load settings. Please try again.');
    }
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
    if (!(checkAuth())) {
      alert('Please log in to save settings.');
      window.location.href = '/login';
      return;
    }
    const formData = new FormData(uiElements.settingsForm);
    const data = Object.fromEntries(formData);
    fetch('/users/me', {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify(data)
    })
      .then(res => {
        if (!res.ok) {
          if (res.status === 401) {
            localStorage.removeItem('token');
            window.location.href = '/login';
          }
          throw new Error('Failed to update settings');
        }
        return res.json();
      })
      .then(() => {
        alert('Settings updated successfully!');
        uiElements.settingsModal.classList.add('hidden');
        toggleSidebar(false);
      })
      .catch(err => {
        console.error('Error updating settings:', err);
        alert('Error updating settings: ' + err.message);
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
           </svg>Show History`
        : `<svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
           </svg>Hide History`;
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
  let pressTimer;
  const handleSendAction = (e) => {
    e.preventDefault();
    if (uiElements.sendBtn.disabled || isRequestActive || isRecording) return;
    if (uiElements.input.value.trim() || uiElements.fileInput.files.length > 0 || uiElements.audioInput.files.length > 0) {
      submitMessage();
    } else {
      pressTimer = setTimeout(() => startVoiceRecording(), 500);
    }
  };

  const handlePressEnd = (e) => {
    e.preventDefault();
    clearTimeout(pressTimer);
    if (isRecording) stopVoiceRecording();
  };

  uiElements.sendBtn.replaceWith(uiElements.sendBtn.cloneNode(true));
  uiElements.sendBtn = document.getElementById('sendBtn');

  uiElements.sendBtn.addEventListener('click', handleSendAction);
  uiElements.sendBtn.addEventListener('touchstart', handleSendAction);
  uiElements.sendBtn.addEventListener('touchend', handlePressEnd);
  uiElements.sendBtn.addEventListener('touchcancel', handlePressEnd);
}

if (uiElements.form) {
  uiElements.form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!isRecording && uiElements.input.value.trim()) {
      submitMessage();
    } else if (!isRecording && (uiElements.fileInput.files.length > 0 || uiElements.audioInput.files.length > 0)) {
      submitMessage();
    }
  });
}

if (uiElements.input) {
  uiElements.input.addEventListener('input', () => {
    updateSendButtonState();
    autoResizeTextarea();
  });
  uiElements.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isRecording && !uiElements.sendBtn.disabled) submitMessage();
    }
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
    if (!(checkAuth())) return alert('Please log in to edit the conversation title.');
    const newTitle = prompt('Enter new conversation title:', currentConversationTitle || '');
    if (newTitle && currentConversationId) {
      updateConversationTitle(currentConversationId, newTitle);
    }
  });
}

if (uiElements.sidebarToggle) {
  uiElements.sidebarToggle.addEventListener('click', () => toggleSidebar());
}

if (uiElements.newConversationBtn) {
  uiElements.newConversationBtn.addEventListener('click', async () => {
    if (!(await checkAuth())) {
      alert('Please log in to create a new conversation.');
      window.location.href = '/login';
      return;
    }
    await createNewConversation();
  });
}

// Debug localStorage
const originalRemoveItem = localStorage.removeItem;
localStorage.removeItem = function (key) {
  console.log('Removing from localStorage:', key);
  originalRemoveItem.apply(this, arguments);
};

// Offline mode detection
window.addEventListener('offline', () => {
  if (uiElements.messageLimitWarning) {
    uiElements.messageLimitWarning.classList.remove('hidden');
    uiElements.messageLimitWarning.textContent = 'You are offline. Some features may be limited.';
  }
});

window.addEventListener('online', () => {
  if (uiElements.messageLimitWarning) {
    uiElements.messageLimitWarning.classList.add('hidden');
  }
});

// Function to auto-resize textarea
function autoResizeTextarea() {
  if (uiElements.input) {
    uiElements.input.style.height = 'auto';
    uiElements.input.style.height = `${uiElements.input.scrollHeight}px`;
    updateSendButtonState();
  }
}

// Function to check if text contains Arabic characters
function isArabicText(text) {
  return /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/.test(text);
}