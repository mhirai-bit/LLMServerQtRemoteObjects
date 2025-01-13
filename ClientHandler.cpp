#include "ClientHandler.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDebug>

ClientHandler::ClientHandler(QWebSocket *socket, QObject *parent)
    : QObject(parent)
    , m_socket(socket)
{
    Q_ASSERT(m_socket);
    // ソケットの親はこの ClientHandler にしない(任意)
    // ただし destructor で手動 close するなら、親にすることも可能

    // ソケットからのシグナル
    connect(m_socket, &QWebSocket::textMessageReceived,
            this, &ClientHandler::onTextMessageReceived);

    connect(m_socket, &QWebSocket::disconnected,
            this, &ClientHandler::onSocketDisconnected);

    // ソケットに何らかのエラーがあればログを出す
    connect(m_socket, QOverload<QAbstractSocket::SocketError>::of(&QWebSocket::errorOccurred),
            this, [this](QAbstractSocket::SocketError error){
                qWarning() << "[ClientHandler] SocketError:" << error << m_socket->errorString();
            });

    // InferenceEngine との接続
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

ClientHandler::~ClientHandler()
{
    if (m_socket) {
        m_socket->close();
        // m_socket->deleteLater(); // 必要に応じて
    }
    qDebug() << "[ClientHandler] Destroyed";
}

void ClientHandler::onTextMessageReceived(const QString &message)
{
    qDebug() << "[ClientHandler] Received text message:" << message;

    // JSONとしてパース
    const QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
    if (!doc.isObject()) {
        qWarning() << "[ClientHandler] Invalid JSON!";
        return;
    }
    QJsonObject obj = doc.object();

    // "action" で処理を分岐
    const QString action = obj.value(QStringLiteral("action")).toString();
    if (action == QLatin1String("generate")) {
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
        m_inference.reinitEngine();

    } else {
        qDebug() << "[ClientHandler] Unknown action:" << action;
    }
}

void ClientHandler::onSocketDisconnected()
{
    qDebug() << "[ClientHandler] onSocketDisconnected";
    emit disconnected(); // サーバー側(QtWSRemoteGenerator)がこれを受けて管理リストから削除し、deleteLater()する
}

//--------------------
// InferenceEngine => クライアントへ送信
//--------------------
void ClientHandler::onPartialResponseReady(const QString &textSoFar)
{
    // JSON {"action":"partialResponse", "content":"..."}
    QJsonObject json;
    json["action"]  = QStringLiteral("partialResponse");
    json["content"] = textSoFar;

    const QByteArray bytes = QJsonDocument(json).toJson(QJsonDocument::Compact);
    m_socket->sendTextMessage(QString::fromUtf8(bytes));
}

void ClientHandler::onGenerationFinished(const QString &finalResponse)
{
    QJsonObject json;
    json["action"]  = QStringLiteral("generationFinished");
    json["content"] = finalResponse;

    const QByteArray bytes = QJsonDocument(json).toJson(QJsonDocument::Compact);
    m_socket->sendTextMessage(QString::fromUtf8(bytes));
}

void ClientHandler::onGenerationError(const QString &errorMessage)
{
    QJsonObject json;
    json["action"]       = QStringLiteral("error");
    json["errorMessage"] = errorMessage;

    const QByteArray bytes = QJsonDocument(json).toJson(QJsonDocument::Compact);
    m_socket->sendTextMessage(QString::fromUtf8(bytes));
}

void ClientHandler::onRemoteInitializedChanged(bool init)
{
    // エンジン初期化の状態が変化したら通知
    QJsonObject json;
    json["action"]     = QStringLiteral("remoteInitializedChanged");
    json["initialized"] = init;

    const QByteArray bytes = QJsonDocument(json).toJson(QJsonDocument::Compact);
    m_socket->sendTextMessage(QString::fromUtf8(bytes));
}
