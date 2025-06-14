# Use a lightweight Nginx image
FROM nginx:alpine

# Remove default Nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy our Docsify documentation files to Nginx's webroot
COPY ../DOCUMENTACAO.md /usr/share/nginx/html/DOCUMENTACAO.md
COPY ../index.html /usr/share/nginx/html/index.html
COPY ../_sidebar.md /usr/share/nginx/html/_sidebar.md

# Nginx by default serves on port 80
EXPOSE 80

# Command to run Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
