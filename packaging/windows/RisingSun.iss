#define MyAppName "Rising Sun"

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

#ifndef SourceDir
  #define SourceDir "dist\\RisingSun"
#endif

#ifndef OutputDir
  #define OutputDir "dist\\installer"
#endif

#ifndef OutputBaseFilename
  #define OutputBaseFilename "RisingSunSetup"
#endif

[Setup]
AppId={{6A3D7D18-5B6E-469D-9A5D-F027F74DA8D1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=Colin Greeley
DefaultDirName={localappdata}\Programs\Rising Sun
DefaultGroupName=Rising Sun
UninstallDisplayIcon={app}\RisingSun.exe
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Rising Sun"; Filename: "{app}\RisingSun.exe"
Name: "{autodesktop}\Rising Sun"; Filename: "{app}\RisingSun.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\RisingSun.exe"; Description: "Launch Rising Sun"; Flags: nowait postinstall skipifsilent