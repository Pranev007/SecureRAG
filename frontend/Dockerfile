# Build the static bundle, then serve it from nginx. The final image contains
# no Node runtime and no source -- just the compiled assets and a web server.

FROM node:22-alpine AS builder

WORKDIR /app

# Copy manifests first so the dependency layer is cached independently of
# source changes.
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY . .

# Vite inlines VITE_* variables at build time, so this must be a build arg
# rather than a runtime environment variable.
ARG VITE_API_BASE_URL=/api/v1
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build


FROM nginx:1.27-alpine AS runtime

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
