#ifndef LLAMA_RESPONSE_GENERATOR_H
#define LLAMA_RESPONSE_GENERATOR_H

#include "rep_LlamaResponseGenerator_source.h"  // Short definitions from .rep file / .repファイルからの定義
#include "llama.h"
#include <QObject>
#include <QString>

/*
  LlamaResponseGenerator:
    - Overrides generate(...) & signals from .rep
    - Uses llama.cpp for text generation
    - Runs model/context init in background

  LlamaResponseGenerator:
    - .repファイルで定義された generate(...) / シグナルをオーバーライド
    - llama.cpp を使用してテキスト生成
    - モデル/コンテキストの初期化をバックグラウンドで実行
*/
class LlamaResponseGenerator : public LlamaResponseGeneratorSimpleSource
{
    Q_OBJECT

public:
    // Constructor: accepts model/context, starts async init
    // コンストラクタ: llama_model/llama_contextを受け取り、非同期初期化を開始
    explicit LlamaResponseGenerator(QObject *parent = nullptr,
                                    llama_model *model = nullptr,
                                    llama_context *ctx = nullptr);

    // Destructor: frees sampler if allocated
    // デストラクタ: samplerがあれば解放
    ~LlamaResponseGenerator() override;

    // Overridden generate(...) from .rep
    // .rep ファイルの generate(...) をオーバーライド
    void generate(const QList<LlamaChatMessage>& messages) override;
    void reinitEngine() override;

signals:
    void reinitialized();

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

    // Runs heavy init in separate thread
    // 別スレッドでの重い初期化処理
    void do_engine_init();

    // Convert QList<LlamaChatMessage> -> std::vector<llama_chat_message>
    // QList<LlamaChatMessage>からstd::vector<llama_chat_message>に変換
    std::vector<llama_chat_message> to_llama_messages(const QList<LlamaChatMessage> &userMessages);
};

#endif // LLAMA_RESPONSE_GENERATOR_H
