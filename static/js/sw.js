// static/js/sw.js
// SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
// SPDX-License-Identifier: Apache-2.0

const CACHE_NAME = 'ChatMG-v2'; // غيّر الإصدار عشان يجبر الـ cache يتحدّث
const urlsToCache = [
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
  '/static/images/icons/mg-512.png',
  'https://raw.githubusercontent.com/4gels/icons/refs/heads/main/splash-screen.png',
  'https://raw.githubusercontent.com/4gels/icons/refs/heads/main/splash-screen-750x1334.png',
  '/static/images/settings-icon.svg',
  '/static/images/swipe-hint.svg'
];

// Install event
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('Opened cache for install');
      return cache.addAll(urlsToCache);
    })
  );
});

// Activate event
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name))
      );
    })
  );
});

// Fetch event with dynamic caching and stale-while-revalidate
self.addEventListener('fetch', (event) => {
  const { url } = event.request;
  const isAPI = url.includes('/api/');
  const isStatic = url.includes('/static/');

  if (isStatic || isAPI) {
    event.respondWith(
      caches.match(event.request).then((response) => {
        if (response) {
          // Return cached version immediately (stale)
          event.waitUntil(updateCache(event.request));
          return response;
        }
        return fetch(event.request).then((fetchResponse) => {
          // Cache the new response
          if (fetchResponse.ok) {
            return caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, fetchResponse.clone());
              return fetchResponse;
            });
          }
          return fetchResponse;
        });
      }).catch(() => {
        // Offline fallback for API
        if (isAPI) {
          return new Response(JSON.stringify({ error: 'Offline mode - Please check your connection' }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
          });
        }
        return caches.match(event.request);
      })
    );
  } else {
    // For other requests, use network-first
    event.respondWith(
      fetch(event.request).catch(() => caches.match(event.request))
    );
  }
});

// Function to update cache in background
async function updateCache(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      await cache.put(request, response.clone());
    }
  } catch (error) {
    console.log('Cache update failed:', error);
  }
}