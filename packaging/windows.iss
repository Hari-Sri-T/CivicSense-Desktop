#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
[Setup]
AppId=org.civicsense.desktop
AppName=CivicSense
AppVersion={#AppVersion}
AppPublisher=CivicSense Project
AppPublisherURL=https://github.com/Hari-Sri-T/CivicSense-Desktop
DefaultDirName={localappdata}\Programs\CivicSense
DefaultGroupName=CivicSense
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\CivicSense.exe
OutputDir={#OutputDir}
OutputBaseFilename=CivicSense-Windows-x64-Setup
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\CivicSense"; Filename: "{app}\CivicSense.exe"
Name: "{autodesktop}\CivicSense"; Filename: "{app}\CivicSense.exe"; Tasks: desktopicon
[Run]
Filename: "{app}\CivicSense.exe"; Description: "Launch CivicSense"; Flags: nowait postinstall skipifsilent
