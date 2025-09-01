// static/js/chat.js
// SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
// SPDX-License-Identifier: Apache-2.0

// Prism.
Prism.plugins.autoloader.languages_path = 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/';

// UI elements.
const chatArea = document.getElementById('chatArea');
const chatBox = document.getElementById('chatBox');
const initialContent = document.getElementById('initialContent');
const form = document.getElementById('footerForm');
const input = document.getElementById('userInput');
const btn = document.getElementById('sendBtn');
const stopBtn = document.getElementById('stopBtn');
const fileBtn = document.getElementById('fileBtn');
const audioBtn = document.getElementById('audioBtn');
const voiceBtn = document.getElementById('voiceBtn');
const fileInput = document.getElementById('fileInput');
const audioInput = document.getElementById('audioInput');
const filePreview = document.getElementById('filePreview');
const audioPreview = document.getElementById('audioPreview');
const promptItems = document.querySelectorAll('.prompt-item');
const mainHeader = document.getElementById('mainHeader');
const chatHeader = document.getElementById('chatHeader');
const homeBtn = document.getElementById('homeBtn');
const clearBtn = document.getElementById('clearBtn');
const loginBtn = document.getElementById('loginBtn');
const messageLimitWarning = document.getElementById('messageLimitWarning');

// Track state.
let streamMsg = null;
let conversationHistory = JSON.parse(sessionStorage.getItem('conversationHistory') || '[]');
let currentAssistantText = '';
let isRequestActive = false;
let abortController = null;
let mediaRecorder = null;
let audioChunks = [];

// تحميل المحادثة عند تحميل الصفحة
document.addEventListener('DOMContentLoaded', () => {
  AOS.init({
    duration: 800,
    easing: 'ease-out-cubic',
    once: true,
    offset: 50,
  });
  if (conversationHistory.length > 0) {
    enterChatView();
    conversationHistory.forEach(msg => {
      addMsg(msg.role, msg.content);
    });
  }
});

// تحقق من الـ token
function checkAuth() {
  return localStorage.getItem('token');
}

// Render markdown content.
function renderMarkdown(el) {
  const raw = el.dataset.text || '';
  const html = marked.parse(raw, {
    gfm: true,
    breaks: true,
    smartLists: true,
    smartypants: false,
    headerIds: false,
  });
  el.innerHTML = '<div class="md-content">' + html + '</div>';
  const wrapper = el.querySelector('.md-content');

  // Wrap tables.
  const tables = wrapper.querySelectorAll('table');
  tables.forEach(t => {
    if (t.parentNode && t.parentNode.classList && t.parentNode.classList.contains('table-wrapper')) return;
    const div = document.createElement('div');
    div.className = 'table-wrapper';
    t.parentNode.insertBefore(div, t);
    div.appendChild(t);
  });

  // Style horizontal rules.
  const hrs = wrapper.querySelectorAll('hr');
  hrs.forEach(h => {
    if (!h.classList.contains('styled-hr')) {
      h.classList.add('styled-hr');
    }
  });

  // Highlight code.
  Prism.highlightAllUnder(wrapper);
}

// Chat view.
function enterChatView() {
  mainHeader.style.display = 'none';
  chatHeader.style.display = 'flex';
  chatHeader.setAttribute('aria-hidden', 'false');
  chatBox.style.display = 'flex';
  initialContent.style.display = 'none';
}

// Home view.
function leaveChatView() {
  mainHeader.style.display = 'flex';
  chatHeader.style.display = 'none';
  chatHeader.setAttribute('aria-hidden', 'true');
  chatBox.style.display = 'none';
  initialContent.style.display = 'flex';
}

// Chat bubble.
function addMsg(who, text) {
  const div = document.createElement('div');
  div.className = 'bubble ' + (who === 'user' ? 'bubble-user' : 'bubble-assist');
  div.dataset.text = text;
  renderMarkdown(div);
  chatBox.appendChild(div);
  chatBox.style.display = 'flex';
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

// Clear all chat.
function clearAllMessages() {
  stopStream(true);
  conversationHistory = [];
  sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
  currentAssistantText = '';
  if (streamMsg) {
    const loadingEl = streamMsg.querySelector('.loading');
    if (loadingEl) loadingEl.remove();
    streamMsg = null;
  }
  chatBox.innerHTML = '';
  input.value = '';
  btn.disabled = true;
  stopBtn.style.display = 'none';
  btn.style.display = 'inline-flex';
  filePreview.style.display = 'none';
  audioPreview.style.display = 'none';
  messageLimitWarning.classList.add('hidden');
  enterChatView();
}

// File preview.
function previewFile() {
  if (fileInput.files.length > 0) {
    const file = fileInput.files[0];
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = e => {
        filePreview.innerHTML = `<img src="${e.target.result}" class="upload-preview">`;
        filePreview.style.display = 'block';
        audioPreview.style.display = 'none';
      };
      reader.readAsDataURL(file);
    }
  }
  if (audioInput.files.length > 0) {
    const file = audioInput.files[0];
    if (file.type.startsWith('audio/')) {
      const reader = new FileReader();
      reader.onload = e => {
        audioPreview.innerHTML = `<audio controls src="${e.target.result}"></audio>`;
        audioPreview.style.display = 'block';
        filePreview.style.display = 'none';
      };
      reader.readAsDataURL(file);
    }
  }
}

// Voice recording.
function startVoiceRecording() {
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.start();
    voiceBtn.classList.add('recording');
    mediaRecorder.addEventListener('dataavailable', event => {
      audioChunks.push(event.data);
    });
  }).catch(err => {
    console.error('Error accessing microphone:', err);
    alert('Failed to access microphone. Please check permissions.');
  });
}

function stopVoiceRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    voiceBtn.classList.remove('recording');
    mediaRecorder.addEventListener('stop', async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const formData = new FormData();
      formData.append('file', audioBlob, 'voice-message.webm');
      submitAudioMessage(formData);
    });
  }
}

// Send audio message.
async function submitAudioMessage(formData) {
  enterChatView();
  addMsg('user', 'Voice message');
  conversationHistory.push({ role: 'user', content: 'Voice message' });
  sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
  streamMsg = addMsg('assistant', '');
  const loadingEl = document.createElement('span');
  loadingEl.className = 'loading';
  streamMsg.appendChild(loadingEl);
  stopBtn.style.display = 'inline-flex';
  btn.style.display = 'none';
  input.value = '';
  btn.disabled = true;
  filePreview.style.display = 'none';
  audioPreview.style.display = 'none';

  isRequestActive = true;
  abortController = new AbortController();

  try {
    const token = checkAuth();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await fetch('/api/audio-transcription', {
      method: 'POST',
      body: formData,
      headers: headers,
      signal: abortController.signal,
    });

    if (!response.ok) {
      if (response.status === 403) {
        messageLimitWarning.classList.remove('hidden');
        input.disabled = true;
        const loadingEl = streamMsg.querySelector('.loading');
        if (loadingEl) loadingEl.remove();
        streamMsg = null;
        isRequestActive = false;
        abortController = null;
        btn.style.display = 'inline-flex';
        stopBtn.style.display = 'none';
        window.location.href = '/login';
        return;
      }
      if (response.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/login';
        return;
      }
      throw new Error('Request failed');
    }

    const data = await response.json();
    const transcription = data.transcription || 'Error: No transcription generated.';
    streamMsg.dataset.text = transcription;
    renderMarkdown(streamMsg);
    streamMsg.dataset.done = '1';
    conversationHistory.push({ role: 'assistant', content: transcription });
    sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));

    streamMsg = null;
    isRequestActive = false;
    abortController = null;
    btn.style.display = 'inline-flex';
    stopBtn.style.display = 'none';
  } catch (error) {
    if (streamMsg) {
      const loadingEl = streamMsg.querySelector('.loading');
      if (loadingEl) loadingEl.remove();
      streamMsg.dataset.text = error.message || 'An error occurred during the request.';
      renderMarkdown(streamMsg);
      streamMsg.dataset.done = '1';
      streamMsg = null;
      isRequestActive = false;
      abortController = null;
    }
    btn.style.display = 'inline-flex';
    stopBtn.style.display = 'none';
  }
}

// Send user message.
async function submitMessage() {
  if (isRequestActive) return;
  let message = input.value.trim();
  let formData = new FormData();
  let endpoint = '/api/chat';
  let inputType = 'text';
  let outputFormat = 'text';

  if (fileInput.files.length > 0) {
    const file = fileInput.files[0];
    if (file.type.startsWith('image/')) {
      endpoint = '/api/image-analysis';
      inputType = 'image';
      message = 'Analyze this image';
      formData.append('file', file);
      formData.append('output_format', 'text');
    }
  } else if (audioInput.files.length > 0) {
    const file = audioInput.files[0];
    if (file.type.startsWith('audio/')) {
      endpoint = '/api/audio-transcription';
      inputType = 'audio';
      message = 'Transcribe this audio';
      formData.append('file', file);
    }
  } else if (message) {
    formData.append('message', message);
    formData.append('system_prompt', 'You are an expert assistant providing detailed, comprehensive, and well-structured responses.');
    formData.append('history', JSON.stringify(conversationHistory));
    formData.append('temperature', '0.7');
    formData.append('max_new_tokens', '128000');
    formData.append('enable_browsing', 'true');
    formData.append('output_format', 'text');
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
  stopBtn.style.display = 'inline-flex';
  btn.style.display = 'none';
  input.value = '';
  btn.disabled = true;
  filePreview.style.display = 'none';
  audioPreview.style.display = 'none';

  isRequestActive = true;
  abortController = new AbortController();

  try {
    const token = checkAuth();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
      headers: headers,
      signal: abortController.signal,
    });

    if (!response.ok) {
      if (response.status === 403) {
        messageLimitWarning.classList.remove('hidden');
        input.disabled = true;
        const loadingEl = streamMsg.querySelector('.loading');
        if (loadingEl) loadingEl.remove();
        streamMsg = null;
        isRequestActive = false;
        abortController = null;
        btn.style.display = 'inline-flex';
        stopBtn.style.display = 'none';
        window.location.href = '/login';
        return;
      }
      if (response.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/login';
        return;
      }
      throw new Error('Request failed');
    }

    if (endpoint === '/api/audio-transcription') {
      const data = await response.json();
      const transcription = data.transcription || 'Error: No transcription generated.';
      streamMsg.dataset.text = transcription;
      renderMarkdown(streamMsg);
      streamMsg.dataset.done = '1';
      conversationHistory.push({ role: 'assistant', content: transcription });
      sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
    } else if (endpoint === '/api/image-analysis') {
      const data = await response.json();
      const analysis = data.image_analysis || 'Error: No analysis generated.';
      streamMsg.dataset.text = analysis;
      renderMarkdown(streamMsg);
      streamMsg.dataset.done = '1';
      conversationHistory.push({ role: 'assistant', content: analysis });
      sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
    } else {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        streamMsg.dataset.text = buffer;
        currentAssistantText = buffer;
        const loadingEl = streamMsg.querySelector('.loading');
        if (loadingEl) loadingEl.remove();
        renderMarkdown(streamMsg);
        chatBox.scrollTop = chatBox.scrollHeight;
      }
      streamMsg.dataset.done = '1';
      conversationHistory.push({ role: 'assistant', content: buffer });
      sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
    }

    streamMsg = null;
    isRequestActive = false;
    abortController = null;
    btn.style.display = 'inline-flex';
    stopBtn.style.display = 'none';
  } catch (error) {
    if (streamMsg) {
      const loadingEl = streamMsg.querySelector('.loading');
      if (loadingEl) loadingEl.remove();
      streamMsg.dataset.text = error.message || 'An error occurred during the request.';
      renderMarkdown(streamMsg);
      streamMsg.dataset.done = '1';
      streamMsg = null;
      isRequestActive = false;
      abortController = null;
    }
    btn.style.display = 'inline-flex';
    stopBtn.style.display = 'none';
  }
}

// Stop streaming and cancel the ongoing request.
function stopStream(forceCancel = false) {
  if (!isRequestActive) return;
  isRequestActive = false;
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
  if (streamMsg && !forceCancel) {
    const loadingEl = streamMsg.querySelector('.loading');
    if (loadingEl) loadingEl.remove();
    streamMsg.dataset.text += '';
    renderMarkdown(streamMsg);
    streamMsg.dataset.done = '1';
    streamMsg = null;
  }
  stopBtn.style.display = 'none';
  btn.style.display = 'inline-flex';
  stopBtn.style.pointerEvents = 'auto';
}

// Prompts.
promptItems.forEach(p => {
  p.addEventListener('click', e => {
    e.preventDefault();
    input.value = p.dataset.prompt;
    submitMessage();
  });
});

// File and audio inputs.
fileBtn.addEventListener('click', () => {
  fileInput.click();
});
audioBtn.addEventListener('click', () => {
  audioInput.click();
});
fileInput.addEventListener('change', previewFile);
audioInput.addEventListener('change', previewFile);

// Voice recording events.
voiceBtn.addEventListener('mousedown', startVoiceRecording);
voiceBtn.addEventListener('touchstart', e => {
  e.preventDefault();
  startVoiceRecording();
});
voiceBtn.addEventListener('mouseup', stopVoiceRecording);
voiceBtn.addEventListener('touchend', e => {
  e.preventDefault();
  stopVoiceRecording();
});

// Submit.
form.addEventListener('submit', e => {
  e.preventDefault();
  submitMessage();
});

// Handle Enter key to submit without adding new line.
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitMessage();
  }
});

// Stop.
stopBtn.addEventListener('click', () => {
  stopBtn.style.pointerEvents = 'none';
  stopStream();
});

// Home.
homeBtn.addEventListener('click', () => {
  window.location.href = '/';
});

// Clear messages.
clearBtn.addEventListener('click', () => {
  clearAllMessages();
});

// Login button.
if (loginBtn) {
  loginBtn.addEventListener('click', () => {
    window.location.href = '/login';
  });
}

// Enable send button only if input has text or files.
input.addEventListener('input', () => {
  btn.disabled = input.value.trim() === '' && fileInput.files.length === 0 && audioInput.files.length === 0;
});
