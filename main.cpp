#include "QtRoRemoteGenerator.h"
#include "QtWSRemoteGenerator.h"
#include <QCoreApplication>

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    qSetMessagePattern("[%{file}:%{line}] %{message}");

    QtRORemoteGenerator llamaResponseGenerator;

    QRemoteObjectHost srcNode(QUrl(QStringLiteral("tcp://0.0.0.0:12345")));
    srcNode.enableRemoting(&llamaResponseGenerator);

    QtWSRemoteGenerator wsRemoteGenerator;
    wsRemoteGenerator.startServer(12346);

    return app.exec();
}
