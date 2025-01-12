#ifndef QTWSREMOTEGENERATOR_H
#define QTWSREMOTEGENERATOR_H

#include <QObject>
#include <QWebSocketServer>
#include <QList>
#include "ClientHandler.h"

/*
  QtWSRemoteGenerator:
    - WebSocketサーバーとして動作し、新規クライアント接続ごとに ClientHandler を生成
    - クライアントとのやり取りは ClientHandler が担当
*/
class QtWSRemoteGenerator : public QObject
{
    Q_OBJECT
public:
    explicit QtWSRemoteGenerator(QObject *parent = nullptr);
    ~QtWSRemoteGenerator();

    bool startServer(quint16 port);

private slots:
    void onNewConnection();

private:
    QWebSocketServer*       m_webSocketServer {nullptr};
    QList<ClientHandler*>   m_clientHandlers; // 接続中のクライアントを保持
};

#endif // QTWSREMOTEGENERATOR_H
