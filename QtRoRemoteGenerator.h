#ifndef QTROREMOTEGENERATOR_H
#define QTROREMOTEGENERATOR_H

#include "rep_LlamaResponseGenerator_source.h"  // Short definitions from .rep file (シンプルソースからのメソッド定義)
#include "InferenceEngine.h"
#include <QObject>
#include <QString>

/*
  QtRORemoteGenerator:
    - Inherits LlamaResponseGeneratorSimpleSource (generated from .rep file)
    - Uses an internal InferenceEngine to handle AI inference
    - Overrides generate(...) and reinitEngine() to delegate to the engine

  QtRORemoteGeneratorクラス:
    - .repファイルから生成されたLlamaResponseGeneratorSimpleSourceを継承
    - 内部でInferenceEngineを使用し、AI推論を処理
    - generate(...), reinitEngine()をオーバーライドし、エンジンに処理を委譲
*/
class QtRORemoteGenerator : public LlamaResponseGeneratorSimpleSource
{
    Q_OBJECT
public:
    /*
      Constructor:
        - Initializes internal InferenceEngine connections
        - Binds engine signals to this class's signals/methods
      コンストラクタ:
        - 内部のInferenceEngineのシグナルを、このクラスのシグナル/メソッドに接続
    */
    explicit QtRORemoteGenerator(QObject *parent = nullptr);

    /*
      Destructor:
        - Default
      デストラクタ:
        - デフォルト
    */
    ~QtRORemoteGenerator() override = default;

    /*
      generate(...):
        - Called when a client requests text generation
        - Delegates to mInferenceEngine
      generate(...):
        - クライアントからテキスト生成要求が来たときに呼ばれる
        - mInferenceEngineに処理を委譲
    */
    void generate(const QList<LlamaChatMessage>& messages) override;

    /*
      reinitEngine():
        - Re-initializes the inference engine
      reinitEngine():
        - 推論エンジンを再初期化
    */
    void reinitEngine() override;

signals:
    /*
      reinitialized():
        - Emitted after the engine has been reinitialized successfully
      reinitialized():
        - エンジンが再初期化に成功した後にemit
    */
    void reinitialized();

private:
    // Internal engine handling inference
    // 推論を処理する内部エンジン
    InferenceEngine mInferenceEngine;
};

#endif // QTROREMOTEGENERATOR_H
