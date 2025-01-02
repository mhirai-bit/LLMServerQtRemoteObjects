#include "llamaresponsegenerator.h"
#include <QCoreApplication>

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    LlamaResponseGenerator llamaResponseGenerator;

    QRemoteObjectHost srcNode(QUrl(QStringLiteral("local:replica")));
    srcNode.enableRemoting(&llamaResponseGenerator);

    return app.exec();
}
