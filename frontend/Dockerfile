# Фронтенд: статические файлы за nginx. Сборка не нужна – обычный HTML, CSS и JavaScript.
FROM nginx:1.27-alpine
COPY index.html app.css app.js config.js /usr/share/nginx/html/
COPY assets /usr/share/nginx/html/assets
# при старте контейнера пишет config.js с адресом API из переменной API_URL
COPY 40-api-url.sh /docker-entrypoint.d/40-api-url.sh
RUN chmod +x /docker-entrypoint.d/40-api-url.sh
EXPOSE 80
