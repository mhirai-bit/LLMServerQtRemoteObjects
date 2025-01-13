#include "qtwsremotegenerator.h"
#include <QDebug>
#include <QtWebSockets/qwebsocketserver.h>

/*
  QtWSRemoteGenerator constructor:
    - Instantiates a QWebSocketServer with the given name/NonSecureMode
    - The server is a child of this class, so it will be cleaned up automatically
  QtWSRemoteGeneratorのコンストラクタ:
    - 指定の名前とNonSecureModeでQWebSocketServerを生成
    - サーバーはこのクラスの子オブジェクトとして管理され、自動的に後始末される
*/
QtWSRemoteGenerator::QtWSRemoteGenerator(QObject *parent)
    : QObject{parent}
    , m_webSocketServer{new QWebSocketServer(QStringLiteral("WS Inference Server"),
                                             QWebSocketServer::NonSecureMode,
                                             this)}
{
}

/*
  QtWSRemoteGenerator destructor:
    - Closes the server if it's listening
    - Deletes all ClientHandler objects in m_clientHandlers
  QtWSRemoteGeneratorのデストラクタ:
    - サーバーがリッスン中ならclose()を呼んで終了
    - m_clientHandlersにあるClientHandlerオブジェクトを全て解放
*/
QtWSRemoteGenerator::~QtWSRemoteGenerator()
{
    if (m_webSocketServer->isListening()) {
        m_webSocketServer->close();
    }
    qDeleteAll(m_clientHandlers);
    m_clientHandlers.clear();
}

/*
  startServer(port):
    - Tries to listen on the specified port
    - If successful, connects the newConnection signal to onNewConnection()
    - Returns true on success, false otherwise
  startServer(port):
    - 指定したポートでのリッスンを試みる
    - 成功した場合はnewConnectionシグナルとonNewConnection()を接続
    - 成功ならtrue、失敗ならfalseを返す
*/
bool QtWSRemoteGenerator::startServer(quint16 port)
{
    const bool ok = m_webSocketServer->listen(QHostAddress::Any, port);
    if (!ok) {
        qWarning() << "[QtWSRemoteGenerator] Failed to listen on port" << port
                   << ":" << m_webSocketServer->errorString();
        return false;
    }
    qDebug() << "[QtWSRemoteGenerator] Listening on port" << port;

    connect(m_webSocketServer, &QWebSocketServer::newConnection,
            this, &QtWSRemoteGenerator::onNewConnection);

    return true;
}

/*
  onNewConnection():
    - Called when a new client connection is detected
    - For each pending connection, create a ClientHandler and store it in m_clientHandlers
    - When ClientHandler signals disconnected, remove it from the list and delete it
  onNewConnection():
    - 新しいクライアント接続が検知された時に呼ばれる
    - 保留中の接続ごとにClientHandlerを生成し、m_clientHandlersに格納
    - ClientHandlerがdisconnectedシグナルを出したらリストから削除し、deleteLater()
*/
void QtWSRemoteGenerator::onNewConnection()
{
    while (m_webSocketServer->hasPendingConnections()) {
        QWebSocket *socket = m_webSocketServer->nextPendingConnection();
        if (!socket) {
            continue;
        }
        qDebug() << "[QtWSRemoteGenerator] New client connected from"
                 << socket->peerAddress().toString() << ":" << socket->peerPort();

        auto *handler = new ClientHandler(socket, this);
        m_clientHandlers.append(handler);

        // Cleanup when the client disconnects
        // クライアントが切断したらクリーンアップ
        connect(handler, &ClientHandler::disconnected,
                this, [this, handler](){
                    m_clientHandlers.removeAll(handler);
                    handler->deleteLater();
                });
    }
}
