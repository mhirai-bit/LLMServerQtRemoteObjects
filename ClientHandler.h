#ifndef CLIENTHANDLER_H
#define CLIENTHANDLER_H

#include <QObject>
#include <QWebSocket>
#include "InferenceEngine.h"

/*
  ClientHandler:
    - Manages communication with a single client (QWebSocket).
    - Receives JSON messages ("generate", "reinit", etc.)
    - Calls InferenceEngine accordingly.
    - Sends back partial/final responses over the socket.
    - Does NOT include QThreadPool or QRunnable directly here.
*/
class ClientHandler : public QObject
{
    Q_OBJECT
public:
    explicit ClientHandler(QWebSocket *socket, QObject *parent = nullptr);
    ~ClientHandler();

signals:
    void disconnected();

private slots:
    void onTextMessageReceived(const QString &message);
    void onSocketDisconnected();

    // InferenceEngine signals -> wrap into JSON and send
    void onPartialResponseReady(const QString &textSoFar);
    void onGenerationFinished(const QString &finalResponse);
    void onGenerationError(const QString &errorMessage);
    void onRemoteInitializedChanged(bool init);

private:
    QWebSocket      *m_socket {nullptr};
    InferenceEngine  m_inference;
};

#endif // CLIENTHANDLER_H
