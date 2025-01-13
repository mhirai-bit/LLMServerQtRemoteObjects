#include "ClientHandler.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDebug>

/*
  Constructor:
    - Sets up connections for both QWebSocket and InferenceEngine
    - Logs socket creation
  コンストラクタ:
    - QWebSocketとInferenceEngine両方のシグナル接続を設定
    - ソケットの生成をログ出力
*/
ClientHandler::ClientHandler(QWebSocket *socket, QObject *parent)
    : QObject(parent)
    , m_socket(socket)
{
    Q_ASSERT(m_socket);

    // Connect signals from the WebSocket
    // WebSocketからのシグナルを接続
    connect(m_socket, &QWebSocket::textMessageReceived,
            this, &ClientHandler::onTextMessageReceived);

    connect(m_socket, &QWebSocket::disconnected,
            this, &ClientHandler::onSocketDisconnected);

    // Log any socket errors
    // ソケットのエラーをログに出す
    connect(m_socket, QOverload<QAbstractSocket::SocketError>::of(&QWebSocket::errorOccurred),
            this, [this](QAbstractSocket::SocketError error){
                qWarning() << "[ClientHandler] SocketError:" << error << m_socket->errorString();
            });

    // Connect signals from the InferenceEngine
    // InferenceEngineのシグナル接続
    connect(&m_inference, &InferenceEngine::partialResponseReady,
            this, &ClientHandler::onPartialResponseReady);
    connect(&m_inference, &InferenceEngine::generationFinished,
            this, &ClientHandler::onGenerationFinished);
    connect(&m_inference, &InferenceEngine::generationError,
            this, &ClientHandler::onGenerationError);
    connect(&m_inference, &InferenceEngine::remoteInitializedChanged,
            this, &ClientHandler::onRemoteInitializedChanged);

    qDebug() << "[ClientHandler] Created for socket" << socket;
}

/*
  Destructor:
    - Closes the socket if it exists
    - Logs destruction
  デストラクタ:
    - ソケットがあればcloseする
    - 破棄をログに出す
*/
ClientHandler::~ClientHandler()
{
    if (m_socket) {
        m_socket->close();
        // m_socket->deleteLater(); // optional / 必要に応じて
    }
    qDebug() << "[ClientHandler] Destroyed";
}

/*
  onTextMessageReceived(message):
    - Called when the client sends a text message
    - Parses JSON and handles "generate" or "reinit" actions
  onTextMessageReceived(message):
    - クライアントからのテキストメッセージを受け取ったときに呼ばれる
    - JSONを解析し、"generate"や"reinit"などのアクションを処理
*/
void ClientHandler::onTextMessageReceived(const QString &message)
{
    qDebug() << "[ClientHandler] Received text message:" << message;

    // Parse as JSON
    // JSONとしてパース
    const QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
    if (!doc.isObject()) {
        qWarning() << "[ClientHandler] Invalid JSON!";
        return;
    }
    QJsonObject obj = doc.object();

    // Check "action" property
    // "action" プロパティで処理を分岐
    const QString action = obj.value(QStringLiteral("action")).toString();
    if (action == QLatin1String("generate")) {
        // Handle "generate"
        // "messages" : Array of { "role":"...", "content":"..." }
        const QJsonArray msgs = obj.value(QStringLiteral("messages")).toArray();
        QList<LlamaChatMessage> messageList;
        for (const QJsonValue &val : msgs) {
            if (!val.isObject()) continue;
            QJsonObject mobj = val.toObject();
            LlamaChatMessage chatMsg;
            chatMsg.setRole(mobj.value(QStringLiteral("role")).toString());
            chatMsg.setContent(mobj.value(QStringLiteral("content")).toString());
            messageList.append(chatMsg);
        }
        m_inference.generate(messageList);

    } else if (action == QLatin1String("reinit")) {
        // Handle "reinit"
        // "reinit" -> calls InferenceEngine's reinitEngine()
        m_inference.reinitEngine();

    } else {
        qDebug() << "[ClientHandler] Unknown action:" << action;
    }
}

/*
  onSocketDisconnected():
    - Called when the WebSocket disconnects
    - Emits disconnected() signal so the parent (server) can remove this handler
  onSocketDisconnected():
    - WebSocketが切断された時に呼ばれる
    - disconnected()シグナルをemitし、親(サーバー)がこのハンドラを削除できるようにする
*/
void ClientHandler::onSocketDisconnected()
{
    qDebug() << "[ClientHandler] onSocketDisconnected";
    emit disconnected();
}

//--------------------
// From InferenceEngine => to Client (via JSON)
// InferenceEngine からのシグナルを JSON 変換してクライアントに送る
//--------------------

/*
  onPartialResponseReady(textSoFar):
    - Sends partial response to the client as JSON with "action":"partialResponse"
  onPartialResponseReady(textSoFar):
    - 部分的な応答をクライアントへ "action":"partialResponse" としてJSON送信
*/
void ClientHandler::onPartialResponseReady(const QString &textSoFar)
{
    QJsonObject json;
    json["action"]  = QStringLiteral("partialResponse");
    json["content"] = textSoFar;

    const QByteArray bytes = QJsonDocument(json).toJson(QJsonDocument::Compact);
    m_socket->sendTextMessage(QString::fromUtf8(bytes));
}

/*
  onGenerationFinished(finalResponse):
    - Sends final generated text to the client as "generationFinished"
  onGenerationFinished(finalResponse):
    - 最終応答を "generationFinished" としてクライアントに送信
*/
void ClientHandler::onGenerationFinished(const QString &finalResponse)
{
    QJsonObject json;
    json["action"]  = QStringLiteral("generationFinished");
    json["content"] = finalResponse;

    const QByteArray bytes = QJsonDocument(json).toJson(QJsonDocument::Compact);
    m_socket->sendTextMessage(QString::fromUtf8(bytes));
}

/*
  onGenerationError(errorMessage):
    - Sends error message to the client as "error"
  onGenerationError(errorMessage):
    - エラーを "error" というアクション名でクライアントへ送信
*/
void ClientHandler::onGenerationError(const QString &errorMessage)
{
    QJsonObject json;
    json["action"]       = QStringLiteral("error");
    json["errorMessage"] = errorMessage;

    const QByteArray bytes = QJsonDocument(json).toJson(QJsonDocument::Compact);
    m_socket->sendTextMessage(QString::fromUtf8(bytes));
}

/*
  onRemoteInitializedChanged(init):
    - Notifies the client about the engine's initialization state
    - "remoteInitializedChanged" => {"initialized": bool}
  onRemoteInitializedChanged(init):
    - エンジンの初期化状態をクライアントに通知
    - "remoteInitializedChanged" => {"initialized": bool}
*/
void ClientHandler::onRemoteInitializedChanged(bool init)
{
    QJsonObject json;
    json["action"]     = QStringLiteral("remoteInitializedChanged");
    json["initialized"] = init;

    const QByteArray bytes = QJsonDocument(json).toJson(QJsonDocument::Compact);
    m_socket->sendTextMessage(QString::fromUtf8(bytes));
}
