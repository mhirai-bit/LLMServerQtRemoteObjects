#ifndef LLAMA_RESPONSE_GENERATOR_H
#define LLAMA_RESPONSE_GENERATOR_H

#include "rep_LlamaResponseGenerator_source.h" // Definitions from .rep file / .repファイル定義
#include "llama.h"
#include <QObject>
#include <QString>

/*
  LlamaResponseGenerator:
  - Implements generate(...) & signals from .rep file
  - Uses llama.cpp for text generation
  - Runs model/context initialization in background

  .repファイルで定義されたgenerate(...)やシグナルを実装
  llama.cppを用いてテキスト生成を行う
  モデル/コンテキストの初期化はバックグラウンドで実行
*/
class LlamaResponseGenerator : public LlamaResponseGeneratorSimpleSource
{
    Q_OBJECT

public:
    // Constructor: takes llama_model/llama_context, starts async init
    // コンストラクタ: llama_model/llama_contextを受け取り、非同期初期化を開始
    explicit LlamaResponseGenerator(QObject *parent = nullptr,
                                    llama_model *model = nullptr,
                                    llama_context *ctx = nullptr);

    // Destructor: frees sampler if allocated
    // デストラクタ: samplerがあれば解放
    ~LlamaResponseGenerator() override;

    // Overridden generate(...) slot from .rep
    // .repファイルで定義されたgenerate(...)スロットをオーバーライド
    void generate(const QList<LlamaChatMessage>& messages) override;

private:
    static constexpr int m_ngl {99};
    static constexpr int m_n_ctx {2048};

    // Default model path (defined via CMake)
    // CMakeで定義されたデフォルトモデルパス
    static const std::string m_model_path;

    // Holds llama params/context/model/sampler
    // llamaパラメータ/コンテキスト/モデル/サンプラーを保持
    llama_model_params m_model_params;
    llama_sampler* m_sampler {nullptr};
    llama_model* m_model {nullptr};
    llama_context_params m_ctx_params;
    llama_context* m_ctx {nullptr};

    // Heavy init in a separate thread (load model, etc.)
    // 別スレッドで重い初期化（モデル読み込みなど）を実行
    void doEngineInit();
    std::vector<llama_chat_message> toLlamaMessages(const QList<LlamaChatMessage> &userMessages);
};

#endif // LLAMA_RESPONSE_GENERATOR_H
