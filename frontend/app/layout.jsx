import './globals.css'

export const metadata = {
  title: 'ApplianceCare AI Assistant',
  description: 'AI-powered appliance repair assistant',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

