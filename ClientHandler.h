#ifndef CLIENTHANDLER_H
#define CLIENTHANDLER_H

#include <QObject>
#include <QWebSocket>
#include "InferenceEngine.h"

/*
  ClientHandler:
    - Handles communication with a single WebSocket client
    - Maintains a WebSocket and an InferenceEngine to exchange JSON messages

  ClientHandlerクラス:
    - 1つのWebSocketクライアントとのやり取りを担当
    - WebSocketとInferenceEngineを保持し、JSON形式のメッセージを送受信する
*/
class ClientHandler : public QObject
{
    Q_OBJECT
public:
    /*
      Constructor: Takes a QWebSocket pointer to handle communication.
      コンストラクタ: QWebSocketポインタを受け取り、通信を担当する。
    */
    explicit ClientHandler(QWebSocket *socket, QObject *parent = nullptr);

    /*
      Destructor: Closes and optionally deletes the socket.
      デストラクタ: ソケットをcloseし、必要に応じて削除する。
    */
    ~ClientHandler();

signals:
    /*
      disconnected:
        - Emitted when the client WebSocket disconnects
        - The server uses this to remove and destroy the handler
      disconnectedシグナル:
        - クライアントのWebSocketが切断されたときにemitされる
        - サーバー側がこれを受けてハンドラをリストから削除/破棄する
    */
    void disconnected();

private slots:
    /*
      onTextMessageReceived(message):
        - Triggered when a text message arrives from the client
        - Parses the JSON and delegates "generate"/"reinit" to InferenceEngine
      onTextMessageReceived(message):
        - クライアントからテキストメッセージを受信したときに呼ばれる
        - JSONを解析し、"generate"/"reinit"といったアクションをInferenceEngineに振り分ける
    */
    void onTextMessageReceived(const QString &message);

    /*
      onSocketDisconnected():
        - Triggered when the WebSocket disconnects
        - Emits disconnected() so the server can clean up
      onSocketDisconnected():
        - WebSocketが切断された時に呼ばれる
        - disconnected()シグナルをemitしてサーバー側でクリーンアップ
    */
    void onSocketDisconnected();

    /*
      onPartialResponseReady(textSoFar):
        - Receives partial inference results from the engine
        - Wraps them in JSON and sends to the client
      onPartialResponseReady(textSoFar):
        - 推論途中経過をエンジンから受け取り
        - JSONでラップしてクライアントに送信する
    */
    void onPartialResponseReady(const QString &textSoFar);

    /*
      onGenerationFinished(finalResponse):
        - Receives final generated response
        - Sends to client as a JSON "generationFinished" message
      onGenerationFinished(finalResponse):
        - エンジンから最終推論結果を受け取り
        - "generationFinished"メッセージとしてクライアントへ送信
    */
    void onGenerationFinished(const QString &finalResponse);

    /*
      onGenerationError(errorMessage):
        - Receives error info from the engine
        - Sends a JSON "error" message to the client
      onGenerationError(errorMessage):
        - エンジンからのエラー情報を受け取り
        - "error"メッセージとしてクライアントへ送信
    */
    void onGenerationError(const QString &errorMessage);

    /*
      onRemoteInitializedChanged(init):
        - Triggered when the remote engine's initialization status changes
        - Sends JSON "remoteInitializedChanged" to notify client
      onRemoteInitializedChanged(init):
        - リモートエンジンの初期化状態が変化したときに呼ばれる
        - "remoteInitializedChanged"メッセージとしてクライアントへ送信
    */
    void onRemoteInitializedChanged(bool init);

private:
    QWebSocket       *m_socket {nullptr};  // The client's WebSocket / クライアントのWebSocket
    InferenceEngine  m_inference;         // Manages AI inference / AI推論を扱うエンジン
};

#endif // CLIENTHANDLER_H
