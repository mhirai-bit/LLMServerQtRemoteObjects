// ================================================================
// InferenceEngine.h
// ================================================================
#ifndef INFERENCEENGINE_H
#define INFERENCEENGINE_H

#include "rep_LlamaResponseGenerator_source.h"  // Short definitions from .rep file / .repファイルからの定義
#include "llama.h"
#include <QObject>
#include <QString>

/*
  InferenceEngine:
    - Manages AI inference using llama.cpp
    - Provides slots to generate text, reinitialize engine
    - Exposes signals for partial/final responses and errors

  InferenceEngineクラス:
    - llama.cppを使ったAI推論を管理
    - テキスト生成やエンジン再初期化のためのスロットを提供
    - 部分/最終レスポンスやエラー用のシグナルを提供
*/
class InferenceEngine : public QObject
{
    Q_OBJECT

    /*
      Q_PROPERTY remoteInitialized:
        - Indicates whether the engine is initialized on the remote side
        - Exposed as a QML property, with getter/setter
      Q_PROPERTY remoteInitialized:
        - リモート側のエンジンが初期化されているかを示す
        - QMLプロパティとして公開（ゲッター/セッター付き）
    */
    Q_PROPERTY(bool remoteInitialized
                   READ remoteInitialized
                       WRITE setRemoteInitialized
                           NOTIFY remoteInitializedChanged
                               FINAL)

public:
    /*
      Constructor:
        - Accepts optional llama_model/llama_context pointers
        - Starts async initialization in a background thread
      コンストラクタ:
        - llama_model/llama_contextポインタを任意で受け取る
        - バックグラウンドスレッドで非同期初期化を開始
    */
    explicit InferenceEngine(QObject *parent = nullptr,
                             llama_model *model = nullptr,
                             llama_context *ctx = nullptr);

    /*
      Destructor:
        - Frees the sampler if allocated
      デストラクタ:
        - サンプラーが確保されていれば解放する
    */
    ~InferenceEngine() override;

    /*
      generate(...):
        - Generates text based on the provided messages
        - Emits partial and final responses during the process
      generate(...):
        - 与えられたメッセージに基づきテキストを生成
        - 推論の途中/最終結果をシグナルで通知
    */
    void generate(const QList<LlamaChatMessage>& messages);

    /*
      reinitEngine():
        - Re-initializes the engine
        - Frees existing model/context/sampler, then reruns do_engine_init()
      reinitEngine():
        - エンジンを再初期化
        - 既存のモデル/コンテキスト/サンプラーを解放し、再度do_engine_init()を実行
    */
    void reinitEngine();

    /*
      remoteInitialized():
        - Getter for mRemoteInitialized property
      remoteInitialized():
        - mRemoteInitializedプロパティのゲッター
    */
    bool remoteInitialized() const;

    /*
      setRemoteInitialized(newRemoteInitialized):
        - Setter for mRemoteInitialized property
        - Emits remoteInitializedChanged if value changes
      setRemoteInitialized(newRemoteInitialized):
        - mRemoteInitializedプロパティのセッター
        - 値が変わった場合はremoteInitializedChangedをemit
    */
    void setRemoteInitialized(bool newRemoteInitialized);

signals:
    /*
      reinitialized():
        - Emitted after a successful reinitEngine()
      reinitialized():
        - reinitEngine()が成功した後にemit
    */
    void reinitialized();

    /*
      partialResponseReady(response):
        - Emitted with partial text during generation
      partialResponseReady(response):
        - テキスト生成途中の部分的なレスポンスをemit
    */
    void partialResponseReady(const QString &response);

    /*
      generationFinished(response):
        - Emitted with final text when generation completes
      generationFinished(response):
        - 生成完了時に最終テキストをemit
    */
    void generationFinished(const QString &response);

    /*
      generationError(error):
        - Emitted if an error occurs during generation
      generationError(error):
        - 生成中にエラーが発生した場合にemit
    */
    void generationError(const QString &error);

    /*
      remoteInitializedChanged(newRemoteInitialized):
        - Emitted when the remoteInitialized property changes
      remoteInitializedChanged(newRemoteInitialized):
        - remoteInitializedプロパティが変更されたときにemit
    */
    void remoteInitializedChanged(bool newRemoteInitialized);

private:
    // Internal parameters
    // 内部パラメータ
    static constexpr int mNGl  {99};
    static constexpr int mNCtx {2048};

    // Default model path (via CMake)
    // デフォルトのモデルパス (CMakeで定義)
    static const std::string mModelPath;

    // Holds llama params/context/model/sampler
    // llama 用パラメータ／コンテキスト／モデル／サンプラーを保持
    llama_model_params mModelParams;
    llama_sampler*     mSampler    {nullptr};
    llama_model*       mModel      {nullptr};
    llama_context_params mCtxParams;
    llama_context*       mCtx      {nullptr};

    bool mRemoteInitialized {false};

    /*
      do_engine_init():
        - Heavy initialization (model/context creation)
        - Runs in a background thread
      do_engine_init():
        - モデル/コンテキストをロードする重い初期化処理
        - バックグラウンドスレッドで実行
    */
    void do_engine_init();

    /*
      to_llama_messages(userMessages):
        - Converts QList<LlamaChatMessage> into std::vector<llama_chat_message>
      to_llama_messages(userMessages):
        - QList<LlamaChatMessage>をstd::vector<llama_chat_message>に変換
    */
    std::vector<llama_chat_message> to_llama_messages(const QList<LlamaChatMessage> &userMessages);

    // ======================
    // Remove "static" usage
    // ======================
    // For each InferenceEngine instance, maintain its own buffer
    // → これでスレッドセーフに（複数エンジンが同時生成しても競合しない）
    std::vector<char> mFormattedBuffer;
    int mPrevLen = 0;
};

#endif // INFERENCEENGINE_H
