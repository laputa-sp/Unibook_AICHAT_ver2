module.exports = {
  apps: [
    {
      name: 'ebook-python-api',
      script: './venv/bin/python',
      args: 'run.py',
      interpreter: 'none',  // Don't use PM2's node interpreter
      cwd: '/workspace/ebook_pdf_ai_chatbot_sp_2025-main',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        ENVIRONMENT: 'production',
        PORT: 3098,
      },
      env_development: {
        ENVIRONMENT: 'development',
        PORT: 3098,
      },
      error_file: './logs/err.log',
      out_file: './logs/out.log',
      log_file: './logs/combined.log',
      time: true,
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    },
  ],
};
