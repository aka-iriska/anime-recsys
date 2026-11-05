#!/bin/sh
# адрес API берётся из переменной окружения API_URL при старте контейнера (nginx выполняет скрипты из /docker-entrypoint.d)
set -eu
printf 'window.API_URL = "%s";\n' "${API_URL:-http://localhost:8000}" > /usr/share/nginx/html/config.js
