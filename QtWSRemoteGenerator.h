#ifndef QTWSREMOTEGENERATOR_H
#define QTWSREMOTEGENERATOR_H

#include <QObject>
#include <QWebSocketServer>
#include <QList>
#include "ClientHandler.h"

/*
  QtWSRemoteGenerator:
    - Operates as a non-secure WebSocket server (NonSecureMode)
    - For each new client connection, creates a ClientHandler
    - Each ClientHandler manages communication with one client

  QtWSRemoteGeneratorクラス (非セキュア版):
    - WebSocketサーバーとして動作 (NonSecureMode)
    - 新規クライアント接続ごとにClientHandlerを生成
    - それぞれのClientHandlerがクライアントとのやり取りを担当
*/
class QtWSRemoteGenerator : public QObject
{
    Q_OBJECT
public:
    /*
      Constructor:
        - Creates a QWebSocketServer in NonSecureMode
        - Parent is set to this object

      コンストラクタ:
        - NonSecureMode でQWebSocketServerを生成
        - 親オブジェクトはthisに設定
    */
    explicit QtWSRemoteGenerator(QObject *parent = nullptr);

    /*
      Destructor:
        - Closes the server if it's listening
        - Deletes all ClientHandler instances

      デストラクタ:
        - サーバーがリッスン中ならクローズ
        - 生成済みのClientHandlerインスタンスを全て削除
    */
    ~QtWSRemoteGenerator();

    /*
      startServer(port):
        - Binds QWebSocketServer to the specified port
        - Returns true if successful, false otherwise

      startServer(port):
        - 指定ポートを使用してQWebSocketServerを起動
        - 成功すればtrue、失敗すればfalseを返す
    */
    bool startServer(quint16 port);

private slots:
    /*
      onNewConnection():
        - Called when a new client connection arrives
        - Creates a ClientHandler for each pending connection

      onNewConnection():
        - 新しいクライアント接続が来た時に呼ばれる
        - 保留中の接続ごとにClientHandlerを作成
    */
    void onNewConnection();

private:
    // The actual WebSocket server instance (NonSecure)
    // 非セキュア版のWebSocketサーバーインスタンス
    QWebSocketServer*       m_webSocketServer {nullptr};

    // List of active ClientHandler objects
    // アクティブなClientHandlerオブジェクトのリスト
    QList<ClientHandler*>   m_clientHandlers;
};

#endif // QTWSREMOTEGENERATOR_H
