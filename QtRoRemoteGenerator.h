#ifndef QTROREMOTEGENERATOR_H
#define QTROREMOTEGENERATOR_H

#include "rep_LlamaResponseGenerator_source.h"  // Short definitions from .rep file / .repファイルからの定義
#include "InferenceEngine.h"
#include <QObject>
#include <QString>

class QtRORemoteGenerator : public LlamaResponseGeneratorSimpleSource
{
    Q_OBJECT
public:
    explicit QtRORemoteGenerator(QObject *parent = nullptr);
    ~QtRORemoteGenerator() override = default;

    void generate(const QList<LlamaChatMessage>& messages) override;
    void reinitEngine() override;

signals:
    void reinitialized();

private:
    InferenceEngine mInferenceEngine;
};

#endif // QTROREMOTEGENERATOR_H
