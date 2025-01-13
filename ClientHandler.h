#ifndef CLIENTHANDLER_H
#define CLIENTHANDLER_H

#include <QObject>
#include <QWebSocket>
#include "InferenceEngine.h"

/*
  ClientHandler:
    - 1つのクライアントとのやり取りを担当
    - WebSocketとInferenceEngineを持ち、JSONメッセージをやり取り
*/
class ClientHandler : public QObject
{
    Q_OBJECT
public:
    explicit ClientHandler(QWebSocket *socket, QObject *parent = nullptr);
    ~ClientHandler();

signals:
    // クライアントとの接続が切れた際、サーバー側が解放するために使うシグナル
    void disconnected();

private slots:
    void onTextMessageReceived(const QString &message);
    void onSocketDisconnected();

    // InferenceEngine からのシグナルを受け取り、JSONにラップして送信
    void onPartialResponseReady(const QString &textSoFar);
    void onGenerationFinished(const QString &finalResponse);
    void onGenerationError(const QString &errorMessage);
    void onRemoteInitializedChanged(bool init);

private:
    QWebSocket       *m_socket {nullptr};
    InferenceEngine  m_inference;
};

#endif // CLIENTHANDLER_H
