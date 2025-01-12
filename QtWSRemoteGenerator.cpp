#include "qtwsremotegenerator.h"
#include <QDebug>
#include <QtWebSockets/qwebsocketserver.h>

QtWSRemoteGenerator::QtWSRemoteGenerator(QObject *parent)
    : QObject{parent}
    , m_webSocketServer{new QWebSocketServer(QStringLiteral("WS Inference Server"),
                                             QWebSocketServer::NonSecureMode,
                                             this)}
{
}

QtWSRemoteGenerator::~QtWSRemoteGenerator()
{
    // サーバー停止
    if (m_webSocketServer->isListening()) {
        m_webSocketServer->close();
    }
    // ClientHandler も解放
    qDeleteAll(m_clientHandlers);
    m_clientHandlers.clear();
}

bool QtWSRemoteGenerator::startServer(quint16 port)
{
    const bool ok = m_webSocketServer->listen(QHostAddress::Any, port);
    if (!ok) {
        qWarning() << "[QtWSRemoteGenerator] Failed to listen on port" << port
                   << ":" << m_webSocketServer->errorString();
        return false;
    }
    qDebug() << "[QtWSRemoteGenerator] Listening on port" << port;

    // 新規接続シグナル
    connect(m_webSocketServer, &QWebSocketServer::newConnection,
            this, &QtWSRemoteGenerator::onNewConnection);

    return true;
}

void QtWSRemoteGenerator::onNewConnection()
{
    // pendingConnection() を取り出し、ClientHandler に委譲
    while (m_webSocketServer->hasPendingConnections()) {
        QWebSocket *socket = m_webSocketServer->nextPendingConnection();
        if (!socket) {
            continue;
        }
        qDebug() << "[QtWSRemoteGenerator] New client connected from"
                 << socket->peerAddress().toString() << ":" << socket->peerPort();

        // ClientHandler を生成し、リストに追加
        auto *handler = new ClientHandler(socket, this);
        m_clientHandlers.append(handler);

        // クライアント切断時のクリーンアップ
        connect(handler, &ClientHandler::disconnected,
                this, [this, handler](){
                    m_clientHandlers.removeAll(handler);
                    handler->deleteLater();
                });
    }
}
