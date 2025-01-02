#ifndef LLAMA_RESPONSE_GENERATOR_H
#define LLAMA_RESPONSE_GENERATOR_H

#include "rep_LlamaResponseGenerator_source.h" // Inherits definitions from the .rep file / .repファイルからの定義を継承
#include "llama.h"
#include <QObject>
#include <QString>

// Inherit from LlamaResponseGeneratorSimpleSource to implement generate(...) and signals declared in the .rep file
// .repファイルで宣言されたgenerate(...)やシグナルを実装するためにLlamaResponseGeneratorSimpleSourceを継承
class LlamaResponseGenerator : public LlamaResponseGeneratorSimpleSource
{
    Q_OBJECT

public:
    // Constructor: accepts llama_model and llama_context
    // コンストラクタ: llama_modelとllama_contextを受け取る
    explicit LlamaResponseGenerator(QObject *parent = nullptr,
                                    llama_model *model = nullptr,
                                    llama_context *ctx = nullptr);

    // Destructor: frees sampler if present
    // デストラクタ: samplerがあれば破棄
    ~LlamaResponseGenerator() override;

    // Override the slot defined in the .rep file (generate(const QString&))
    // .repファイルで定義されたスロット(generate(const QString&))をオーバーライド
    void generate(const QString &request) override;

private:
    // Stores llama_context / llama_model / llama_sampler
    // llama_context / llama_model / llama_samplerを保持
    llama_model   *m_model   { nullptr };
    llama_context *m_ctx     { nullptr };
    llama_sampler *m_sampler { nullptr };

    // Initialization before text generation
    // テキスト生成前の初期化処理
    void initializeSampler();
};

#endif // LLAMA_RESPONSE_GENERATOR_H
