import { ModalProvider } from "../../hooks/contexts/Modal";
import NavBar from "../organisms/NavBar";
import { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export function AppLayout({ children }: Props) {
  return (
    <ModalProvider>
      <div className="md:flex md:flex-row">
        <NavBar />
        <div className="h-full w-full">{children}</div>
      </div>
    </ModalProvider>
  );
}
