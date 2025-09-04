// static/js/sw.js
// SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
// SPDX-License-Identifier: Apache-2.0

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open('ChatMG-v1').then((cache) => {
      return cache.addAll([
        '/',
        '/chat',
        '/static/css/style.css',
        '/static/css/chat/style.css',
        '/static/css/sidebar.css',
        '/static/js/chat.js',
        '/static/images/mg.svg',
        '/static/images/icons/mg-48.png',
        '/static/images/icons/mg-72.png',
        '/static/images/icons/mg-96.png',
        '/static/images/icons/mg-128.png',
        '/static/images/icons/mg-192.png',
        '/static/images/icons/mg-256.png',
        '/static/images/icons/mg-384.png',
        '/static/images/icons/mg-512.png'
      ]);
    })
  );
});
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.filter((name) => name !== 'ChatMG-v1').map((name) => caches.delete(name))
      );
    })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
