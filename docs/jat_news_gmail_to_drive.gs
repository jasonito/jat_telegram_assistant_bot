/**
 * Google Apps Script Web App for exporting JAT News and HOUSE News emails to Drive.
 *
 * Deployment:
 * 1. Create a Google Apps Script project.
 * 2. Paste this file.
 * 3. Set Script property NEWS_EXPORT_WEBHOOK_SECRET to the same value as
 *    NEWS_EXPORT_WEBHOOK_SECRET in .env.main.
 * 4. Deploy as Web app and set access to "Anyone with the link".
 * 5. Put the Web app /exec URL in NEWS_EXPORT_WEBHOOK_URL.
 */

const CONFIG = {
  processedLabel: 'to_drive/processed',
  folderName: 'daily_news',
  maxThreadsPerRun: 20,
  saveAttachments: true,
  webhookSecret: 'jat-news-0973'
};

const EXPORT_PROFILES = {
  export_jat_news: {
    query: 'subject:"[JAT News]" newer_than:3d',
    fileSuffix: 'Jat_News',
    subfolderName: 'jat_news'
  },
  export_house_news: {
    query: 'subject:"[HOUSE News]" newer_than:3d',
    fileSuffix: 'HOUSE_News',
    subfolderName: 'house_news'
  }
};

function doPost(e) {
  const payload = parseJsonBody_(e);
  const expectedSecret =
    PropertiesService.getScriptProperties().getProperty('NEWS_EXPORT_WEBHOOK_SECRET') ||
    CONFIG.webhookSecret ||
    '';

  if (!expectedSecret) {
    return jsonResponse_({ ok: false, error: 'missing_secret' }, 500);
  }
  if (!payload || payload.secret !== expectedSecret) {
    return jsonResponse_({ ok: false, error: 'unauthorized' }, 403);
  }

  const profile = EXPORT_PROFILES[payload.action];
  if (!profile) {
    return jsonResponse_({ ok: false, error: 'unknown_action', action: payload.action || '' }, 400);
  }

  const result = exportEmailsToDrive_(profile);
  return jsonResponse_({ ok: true, action: payload.action, result }, 200);
}

function exportEmailsToDrive_(profile) {
  let processedLabel = GmailApp.getUserLabelByName(CONFIG.processedLabel);
  if (!processedLabel) {
    processedLabel = GmailApp.createLabel(CONFIG.processedLabel);
  }

  const rootFolder = getRequiredFolderByName_(CONFIG.folderName);
  const folder = getOrCreateChildFolder_(rootFolder, profile.subfolderName);
  const threads = GmailApp.search(profile.query, 0, CONFIG.maxThreadsPerRun);

  let writtenFiles = 0;
  let processedThreads = 0;
  let skippedThreads = 0;

  threads.forEach(thread => {
    const threadLabelNames = thread.getLabels().map(l => l.getName());
    if (threadLabelNames.includes(CONFIG.processedLabel)) {
      skippedThreads += 1;
      return;
    }

    let wroteSomething = false;
    const messages = thread.getMessages();

    messages.forEach(message => {
      const dateObj = message.getDate();
      const dateStr = Utilities.formatDate(dateObj, Session.getScriptTimeZone(), 'yyyy-MM-dd');
      const subjectRaw = message.getSubject() || 'No Subject';
      const from = message.getFrom() || '';
      const to = message.getTo() || '';
      const cc = message.getCc() || '';
      const dateIso = dateObj.toISOString();
      const messageId = message.getId();
      const body = message.getPlainBody() || '';

      const content = [
        `# ${subjectRaw}`,
        `**From:** ${from}  `,
        `**To:** ${to}  `,
        `**Cc:** ${cc}  `,
        `**Date:** ${dateIso}  `,
        `**Message ID:** \`${messageId}\``,
        '',
        '---',
        '## Email Body',
        '',
        body
      ].join('\n');

      const fileName = `${dateStr}_${profile.fileSuffix}.md`;
      if (!fileExists_(folder, fileName)) {
        folder.createFile(fileName, content, MimeType.PLAIN_TEXT);
        wroteSomething = true;
        writtenFiles += 1;
      }

      if (CONFIG.saveAttachments) {
        const attachments = message.getAttachments({ includeInlineImages: false });
        attachments.forEach((attachment, index) => {
          const originalName = attachment.getName() || `attachment_${index + 1}`;
          const attName = sanitizeFileName_(originalName);
          const attachmentFileName = `${dateStr}_${profile.fileSuffix}_att_${index + 1}_${attName}`;

          if (!fileExists_(folder, attachmentFileName)) {
            folder.createFile(attachment.copyBlob().setName(attachmentFileName));
            wroteSomething = true;
            writtenFiles += 1;
          }
        });
      }
    });

    if (wroteSomething) {
      thread.addLabel(processedLabel);
      processedThreads += 1;
    }
  });

  return {
    matchedThreads: threads.length,
    processedThreads,
    skippedThreads,
    writtenFiles,
    folderName: CONFIG.folderName,
    subfolderName: profile.subfolderName,
    query: profile.query,
    fileSuffix: profile.fileSuffix
  };
}

function organizeExistingDailyNewsFiles() {
  const rootFolder = getRequiredFolderByName_(CONFIG.folderName);
  const jatFolder = getOrCreateChildFolder_(rootFolder, EXPORT_PROFILES.export_jat_news.subfolderName);
  const houseFolder = getOrCreateChildFolder_(rootFolder, EXPORT_PROFILES.export_house_news.subfolderName);

  let movedJat = 0;
  let movedHouse = 0;
  const files = rootFolder.getFiles();
  while (files.hasNext()) {
    const file = files.next();
    const name = file.getName();
    if (/_Jat_News(?:_att_\d+_.*)?\.md$/i.test(name) || /_Jat_News_att_\d+_/i.test(name)) {
      moveFileToFolder_(file, rootFolder, jatFolder);
      movedJat += 1;
    } else if (/_HOUSE_News(?:_att_\d+_.*)?\.md$/i.test(name) || /_HOUSE_News_att_\d+_/i.test(name)) {
      moveFileToFolder_(file, rootFolder, houseFolder);
      movedHouse += 1;
    }
  }

  return { ok: true, movedJat, movedHouse };
}

function getRequiredFolderByName_(folderName) {
  const folders = DriveApp.getFoldersByName(folderName);
  if (!folders.hasNext()) {
    throw new Error(`Drive folder not found: ${folderName}. Please create it first.`);
  }
  return folders.next();
}

function getOrCreateChildFolder_(parentFolder, childName) {
  const folders = parentFolder.getFoldersByName(childName);
  if (folders.hasNext()) {
    return folders.next();
  }
  return parentFolder.createFolder(childName);
}

function moveFileToFolder_(file, oldFolder, newFolder) {
  if (fileExists_(newFolder, file.getName())) {
    oldFolder.removeFile(file);
    return;
  }
  newFolder.addFile(file);
  oldFolder.removeFile(file);
}

function parseJsonBody_(e) {
  try {
    return JSON.parse((e && e.postData && e.postData.contents) || '{}');
  } catch (err) {
    return null;
  }
}

function jsonResponse_(body, statusCode) {
  return ContentService
    .createTextOutput(JSON.stringify({ ...body, statusCode }))
    .setMimeType(ContentService.MimeType.JSON);
}

function fileExists_(folder, fileName) {
  return folder.getFilesByName(fileName).hasNext();
}

function sanitizeFileName_(name) {
  return String(name)
    .replace(/[\\/:*?"<>|#\[\]]/g, '_')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 150);
}
