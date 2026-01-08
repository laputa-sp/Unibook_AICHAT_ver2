#!/bin/bash
# PM2 관리 명령어 모음

case "$1" in
  start)
    echo "🚀 Python 서버 시작..."
    pm2 start ecosystem.config.js
    pm2 logs ebook-python-api --lines 50
    ;;

  stop)
    echo "🛑 Python 서버 중지..."
    pm2 stop ebook-python-api
    ;;

  restart)
    echo "🔄 Python 서버 재시작..."
    pm2 restart ebook-python-api
    pm2 logs ebook-python-api --lines 30
    ;;

  status)
    echo "📊 서버 상태 확인..."
    pm2 status
    pm2 info ebook-python-api
    ;;

  logs)
    echo "📝 로그 확인..."
    pm2 logs ebook-python-api
    ;;

  monit)
    echo "📈 실시간 모니터링..."
    pm2 monit
    ;;

  delete)
    echo "🗑️  서버 삭제..."
    pm2 delete ebook-python-api
    ;;

  save)
    echo "💾 현재 상태 저장..."
    pm2 save
    echo "✅ 부팅 시 자동 시작 설정됨"
    ;;

  *)
    echo "📖 사용법:"
    echo "  ./pm2-commands.sh start      # 서버 시작"
    echo "  ./pm2-commands.sh stop       # 서버 중지"
    echo "  ./pm2-commands.sh restart    # 서버 재시작"
    echo "  ./pm2-commands.sh status     # 상태 확인"
    echo "  ./pm2-commands.sh logs       # 로그 보기"
    echo "  ./pm2-commands.sh monit      # 실시간 모니터링"
    echo "  ./pm2-commands.sh delete     # 서버 삭제"
    echo "  ./pm2-commands.sh save       # 부팅 시 자동 시작 설정"
    ;;
esac
