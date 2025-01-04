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

private:
    // Internal parameters
    // 内部パラメータ
    static constexpr int m_n_gl  {99};
    static constexpr int m_n_ctx {2048};

    // Default model path (via CMake)
    // デフォルトのモデルパス (CMakeで定義)
    static const std::string m_model_path;

    // Holds llama params/context/model/sampler
    // llama 用パラメータ／コンテキスト／モデル／サンプラーを保持
    llama_model_params m_model_params;
    llama_sampler*     m_sampler     {nullptr};
    llama_model*       m_model       {nullptr};
    llama_context_params m_ctx_params;
    llama_context*       m_ctx       {nullptr};

    // Runs heavy init in separate thread
    // 別スレッドでの重い初期化処理
    void do_engine_init();

    // Convert QList<LlamaChatMessage> -> std::vector<llama_chat_message>
    // QList<LlamaChatMessage>からstd::vector<llama_chat_message>に変換
    std::vector<llama_chat_message> to_llama_messages(const QList<LlamaChatMessage> &user_messages);
};

#endif // LLAMA_RESPONSE_GENERATOR_H
